import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ProductionEnv(gym.Env):
    """
    Gymnasium-compatible RL environment for production planning.

    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_products=6,
        n_raw_types=4,
        max_inventory=30,
        max_raw_inventory=30,
        max_demand=30,
        max_buy=30,
        raw_capacity=50,
        episode_length=12,
        sales_price=5,
        production_cost=3,
        new_raw_costs=None,        # array of length n_raw_types
        recycled_raw_costs=None,   # array of length n_raw_types
        holding_cost=0.1,
        production_coeffs=None,    # shape (n_products, n_raw_types)
        can_be_recycled=None,      # array of bool, shape (n_raw_types,)
    ):    
        super().__init__()
        self.n_products = n_products
        self.n_raw_types = n_raw_types
        # Only some raw types can be replaced by recycled
        if can_be_recycled is None:
            self.can_be_recycled = np.array([1]* (n_raw_types // 2) + [0]* (n_raw_types - n_raw_types // 2), dtype=bool)
        else:
            self.can_be_recycled = np.array(can_be_recycled, dtype=bool)

        

        self.nb_sec_factors = np.count_nonzero(self.can_be_recycled)  
        self.max_inventory = max_inventory
        self.max_raw_inventory = max_raw_inventory
        self.max_demand = max_demand
        self.max_buy = max_buy
        self.raw_capacity = raw_capacity
        self.sales_price = [np.random.randint(120, 150) for _ in range(n_products)]
        self.production_cost = [np.random.randint(10, 20) for _ in range(n_products)]
        self.new_raw_costs = [np.random.randint(30, 60) for _ in range(self.n_raw_types-self.nb_sec_factors)]
        self.recycled_raw_costs = [np.random.randint(5, 10) for i in range(self.nb_sec_factors)]
        self.holding_cost = [np.random.uniform(0.5, 2.5) for _ in range(n_products)]
        self.episode_length = episode_length
        self.demand_covering = []
        self.total_delivered = [0 for _ in range(n_products)]
        self.total_demand = [0 for _ in range(n_products)]
        '''self.production_coeffs = (
            np.ones((n_products, n_raw_types), dtype=int)
            if production_coeffs is None
            else np.array(production_coeffs)
        )'''

        '''A: availability of secondary material,
          I_minus_I_A: set of primary prod factor, 
          R_fix: capacity of primary prod factor
          '''
        self.Avl = [
            [np.random.randint(15, 35) for _ in range(self.episode_length)] 
            for i, flag in enumerate(self.can_be_recycled) if flag
        ]

        self.production_coeffs = [[np.random.randint(0, 8) for _ in range(n_products)] for _ in range(n_raw_types)]
        self.R_fix = [
            [np.random.randint(50, 100) for _ in range(self.episode_length)]
            for i, flag in enumerate(self.can_be_recycled) if not flag
        ]
        
        # Action: [produce_1..n, buy_new_raw_1..m, buy_recycled_raw_1..m, sell_1..n]
        self.action_space = spaces.MultiDiscrete(
            [self.max_inventory] * self.n_products +    # produce
            [self.max_buy] * (self.n_raw_types - self.nb_sec_factors) +         # buy new
            [self.max_buy] * self.nb_sec_factors +         # buy recycled
            [self.max_demand] * self.n_products         # sell
        )

        # Observation: [final_inventory_1..n, raw_inventory_1..m]
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.max_inventory, self.max_raw_inventory, self.raw_capacity),
            shape=(self.n_products + self.n_raw_types - self.nb_sec_factors),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.final_inventory = np.random.randint(3, 5, size=self.n_products)
        self.raw_inventory = np.random.randint(10, 50, size=self.n_raw_types)
        self.timestep = 0
        self.total_delivered = [0 for _ in range(self.n_products)]
        self.total_demand = [0 for _ in range(self.n_products)]
        return self._get_obs(), {}

    def step(self, action):

        action = np.array(action)
        produce = np.clip(action[:self.n_products], 0, self.max_inventory - self.final_inventory)
        buy_new_raw = np.clip(action[self.n_products:self.n_products+self.n_raw_types], 0, self.raw_capacity - self.raw_inventory)
        buy_recycled_raw =action[self.n_products+self.n_raw_types:self.n_products+2*self.n_raw_types],
        sell = action[self.n_products+2*self.n_raw_types:],
            
        # Generate availability for recyclable raw types
        availability = np.zeros(self.n_raw_types, dtype=int)
        for i, flag in enumerate(self.can_be_recycled):
            if flag:
                availability[i] = np.random.randint(15, 35) 
            

        # Only allow recycled for types that can be recycled
        actual_buy_recycled_raw = np.where(
            self.can_be_recycled,
            np.minimum(
                np.clip(buy_recycled_raw, 0, availability),
                self.max_raw_inventory - self.raw_inventory
            ),
            0
        )

        # Respect max_raw_inventory
        actual_buy_new_raw = np.minimum(
            buy_new_raw, self.max_raw_inventory - (self.raw_inventory + actual_buy_recycled_raw)
            )
        
        # Unified inventory: add both new and recycled purchases for recyclable types, only new for others
        self.raw_inventory = np.minimum(
            self.raw_inventory + actual_buy_new_raw + actual_buy_recycled_raw,
            self.max_raw_inventory
        )

        # Production
        actual_produced = np.zeros(self.n_products, dtype=int)
        for i in range(self.n_products):
            max_by_raw = self.raw_inventory // self.production_coeffs[i]
            max_possible = min(produce[i], np.min(max_by_raw))
            self.raw_inventory -= max_possible * self.production_coeffs[i]
            self.final_inventory[i] += max_possible
            actual_produced[i] = max_possible

        sum_over_products = sum(self.production_coeffs[j][i] * actual_produced[i] for i in range(self.n_products))
        assert sum_over_products <= self.raw_inventory[j][t]

        # Simulate demand
        demand = np.random.randint(10, self.max_demand, size=self.n_products)
        actual_sold = np.minimum(sell, np.minimum(self.final_inventory, demand))
        self.final_inventory -= actual_sold

        self.total_delivered += np.sum(actual_sold)
        # Reward calculation
        reward = (
            np.sum(actual_sold * self.sales_price)
            - np.sum(produce * self.production_cost)
            - np.sum(actual_buy_new_raw * self.new_raw_costs)
            - np.sum(actual_buy_recycled_raw * self.recycled_raw_costs)
            - self.holding_cost * (np.sum(self.final_inventory) + np.sum(self.raw_inventory))
        )

        self.timestep += 1
        terminated = self.timestep >= self.episode_length
        truncated = False

        self.last_action = action
        self.last_produced = actual_produced
        self.last_buy_new_raw = actual_buy_new_raw
        self.last_buy_recycled_raw = actual_buy_recycled_raw
        self.demand_covering = [demand, actual_sold] #first is demand, second is quantity sold
        self.last_demand = demand
        self.last_units_sold = actual_sold
        self.last_reward = reward

        return self._get_obs(), reward, terminated, truncated, {}


    def _get_obs(self):
        # [final_inventory_1..n, raw_inventory_1..m]
        return np.concatenate([self.final_inventory, self.raw_inventory]).astype(np.float32)

    def render(self):
        print(
             f"Timestep: {self.timestep}\n"
            f"Action taken (produce, buy_new_raw, buy_recycled_raw, sell):\n{self.last_action}\n"
            f"Produced: {self.last_produced}\n"
            f"Bought new raw: {self.last_buy_new_raw}\n"
            f"Bought recycled raw: {self.last_buy_recycled_raw}\n"
            f"Demand: {self.last_demand}\n"
            f"Units sold: {self.last_units_sold}\n"
            f"Final inventory: {self.final_inventory}\n"
            f"Recycled raw inventory: {self.raw_inventory}\n"
            f"Reward: {self.last_reward}\n"
        )

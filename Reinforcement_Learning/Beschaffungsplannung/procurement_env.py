import numpy as np
import math

class procurement_env:
    def __init__(
        self,
        max_inventory,
        max_demand,
        max_availability,
        unit_procurement_cost=5.0,
        unit_holding_cost=1.0,
        fixed_procurement_cost=5.0,
        unit_storage_cost=20.0,
        stockout_cost=2.0,
        episode_length=20,
        demand_binom_p=0.5,
        availability_binom_p=0.5,
        seed=None,
    ):
        self.max_inventory = max_inventory
        self.max_demand = max_demand
        self.max_availability = max_availability
        self.unit_procurement_cost = unit_procurement_cost
        self.unit_holding_cost = unit_holding_cost
        self.fixed_procurement_cost = fixed_procurement_cost
        self.unit_storage_cost = unit_storage_cost
        self.stockout_cost = stockout_cost
        self.episode_length = episode_length
        self.demand_binom_p = demand_binom_p
        self.availability_binom_p = availability_binom_p
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        #self.inventory = self.rng.integers(0, self.max_inventory + 1)
        self.inventory = 10
        self.timestep = 0
        self.stockout_events = 0
        self.history = []
        return self.inventory

    def get_action_space(self, availability):
        # Actions: 0 to min(availability, max_inventory - inventory)
        max_order = min(availability, self.max_inventory - self.inventory)
        return np.arange(0, max_order + 1)

    def step(self, action):
        # Sample demand and availability
        #demand = self.rng.binomial(self.max_demand, self.demand_binom_p)
        demand = 2 
        #availability = self.rng.binomial(self.max_availability, self.availability_binom_p)
        availability = 8

        # Clip action to feasible range
        max_order = min(availability, self.max_inventory - self.inventory)
        order_qty = int(np.clip(action, 0, max_order))

        # Update inventory
        prev_inventory = self.inventory
        self.inventory += order_qty
        self.inventory -= demand

        # Costs
        procurement_cost = order_qty * self.unit_procurement_cost
        fixed_cost = self.fixed_procurement_cost if order_qty > 0 else 0
        holding_cost = max(0, self.inventory) * self.unit_holding_cost
        storage_cost = max(0, self.inventory) * self.unit_storage_cost
        stockout = -min(0, self.inventory)
        stockout_cost = stockout * self.stockout_cost

        total_cost = procurement_cost + fixed_cost + holding_cost + storage_cost + stockout_cost
        reward = -total_cost

        # Track stockout event
        stockout_event = int(self.inventory < 0)
        self.stockout_events += stockout_event


        # Save history for analysis
        self.history.append({
            "inventory": prev_inventory,
            "order": order_qty,
            "demand": demand,
            "availability": availability,
            "cost": total_cost,
            "stockout": stockout,
            "stockout_event": stockout_event,
        })

        self.timestep += 1
        done = self.timestep >= self.episode_length

        info = {
            "stockout_event": stockout_event,
            "stockout_probability": self.stockout_events / self.timestep,
            "demand": demand,
            "availability": availability,
            "order": order_qty,
            "inventory": self.inventory,
            "cost": total_cost,
        }

        # State is the current inventory, clipped to allowed range
        state = int(np.clip(self.inventory, -self.max_demand, self.max_inventory))
        return state, reward, done, info

    def sample_demand(self):
        return self.rng.binomial(self.max_demand, self.demand_binom_p)
        #return math.comb(self.max_demand, d) * (self.demand_binom_p ** d) * ((1 - self.demand_binom_p) ** (self.max_demand - d))

    def sample_availability(self):
        return self.rng.binomial(self.max_availability, self.availability_binom_p)
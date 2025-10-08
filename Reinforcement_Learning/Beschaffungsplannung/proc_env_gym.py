import gymnasium as gym
from gymnasium import spaces
import numpy as np
from procurement_env import procurement_env

class GymProcurementEnv(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.env = procurement_env(**kwargs)
        # Observation: inventory level (discrete)
        self.observation_space = spaces.Box(
            low=-self.env.max_demand, high=self.env.max_inventory, shape=(1,), dtype=np.int32
        )
        # Action: order quantity (discrete, but we use Box for SB3 compatibility)
        self.action_space = spaces.Discrete(self.env.max_inventory + 1)
        self._last_availability = None

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return np.array([obs], dtype=np.int32), {}

    def step(self, action):
        # Sample availability for this step
        availability = self.env.sample_availability()
        # Clip action to feasible range
        max_order = min(availability, self.env.max_inventory - self.env.inventory)
        action = int(np.clip(action, 0, max_order))
        obs, reward, done, info = self.env.step(action)
        return np.array([obs], dtype=np.int32), reward, done, False, info

    def render(self):
        print(f"Inventory: {self.env.inventory}, Timestep: {self.env.timestep}")
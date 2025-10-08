import gymnasium as gym
from stable_baselines3 import PPO
from env import ProductionEnv
import numpy as np
import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env = ProductionEnv()


vec_env = make_vec_env(lambda: env, n_envs=1)

def init_weights_glorot(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)   # Glorot-Uniform
        nn.init.zeros_(m.bias)
        

# Define the policy architecture
policy_kwargs = dict(
    activation_fn=nn.Tanh,                 # tanh activation
    net_arch=[256, 256] if int(np.prod(env.action_space.nvec)) < 1000 else [512, 512],  # layers
)

# Instantiate the agent
model = PPO("MlpPolicy",
    vec_env,
    learning_rate=1e-3,
    n_steps=256,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    n_epochs=10,
    policy_kwargs= policy_kwargs,
    verbose=1)


# Apply after creating model
model.policy.apply(init_weights_glorot)

# Train the agent
model.learn(total_timesteps=1_000_000)

# Save the model
model.save("ppo_basic_production")

# Test the trained agent
obs, _ = env.reset()
done = False
total_reward = 0

n_products = env.n_products
n_complet_served_periods = np.zeros(n_products)
demand_covering = []

while not done:
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    demand_covering.append(env.demand_covering) 
    total_reward += reward
    done = terminated or truncated
print(f"Total reward (PPO): {total_reward}")

print(n_products)
for step in range(len(demand_covering)):
    for n in range(n_products):
        if demand_covering[step][0][n] == demand_covering[step][1][n]:
            n_complet_served_periods[n] += 1

print("Number of completely served periods per product:", n_complet_served_periods/len(demand_covering))
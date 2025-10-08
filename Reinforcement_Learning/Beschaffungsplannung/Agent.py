import torch
import torch.optim as optim
import numpy as np
from proc_env_gym import GymProcurementEnv
from CostApproximator import CostApproximatorNN

# Hyperparameters
episodes = 1000
steps_per_episode = 20
batch_size = 64
learning_rate = 1e-3

# Create environment and network
env = GymProcurementEnv(
    max_inventory=20,
    max_demand=10,
    max_availability=15,
    unit_procurement_cost=2.0,
    unit_holding_cost=0.1,
    fixed_procurement_cost=5.0,
    unit_storage_cost=0.05,
    stockout_cost=10.0,
    episode_length=steps_per_episode,
    demand_binom_p=0.5,
    availability_binom_p=0.7,
    seed=42,
)
net = CostApproximatorNN()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

# Collect experience
states, actions, costs, next_state= [], [], [], []
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        # Random action for exploration
        availability = env.env.sample_availability()
        max_order = min(availability, env.env.max_inventory - env.env.inventory)
        action = np.random.randint(0, max_order + 1)
        next_obs, reward, done, _, info = env.step(action)
        # Store (state, action, cost)
        states.append([obs[0]])
        actions.append([action])
        next_state.append([next_obs[0]])
        costs.append([-reward])  # reward is negative cost
        obs = next_obs

# Convert to tensors
states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.float32)
next_states = torch.tensor(next_state, dtype=torch.float32)
#costs = torch.tensor(costs, dtype=torch.float32)

# Train the network
# For cost prediction:
'''for epoch in range(100):
    idx = np.random.choice(len(states), batch_size)
    batch_states = states[idx]
    batch_actions = actions[idx]
    batch_costs = costs[idx]
    pred = net(batch_states, batch_actions)
    loss = loss_fn(pred, batch_costs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("Training complete.")'''

# for next state prediction:
for epoch in range(100):
    idx = np.random.choice(len(states), batch_size)
    batch_states = states[idx]
    batch_actions = actions[idx]
    batch_next_states = next_states[idx]
    pred = net(batch_states, batch_actions)
    loss = loss_fn(pred, batch_next_states)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("Training complete.")

print(net(-10,8))
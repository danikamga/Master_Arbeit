import torch
import torch.nn as nn

class CostApproximatorNN(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        # state, action: tensors of shape (batch, 1)
        state=torch.tensor([[state]], dtype=torch.float32) if isinstance(state, (int, float)) else state
        action=torch.tensor([[action]], dtype=torch.float32) if isinstance(action, (int, float)) else action
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
    
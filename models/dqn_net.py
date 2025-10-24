# src/models/dqn_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(128, 128)):
        super().__init__()
        if isinstance(hidden, int):
            hidden = [hidden]
        
        layers = []
        last_dim = state_dim
        for h in hidden:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # Accept numpy arrays or torch tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)
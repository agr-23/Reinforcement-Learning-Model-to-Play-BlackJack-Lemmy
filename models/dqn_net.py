# src/models/dqn_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNet(nn.Module):
    """
    Deep Q-Network for Blackjack (Lemmy)
    - Enhanced capacity
    - Suitable for Double DQN training
    """

    def __init__(self, state_dim, action_dim, hidden=(256, 256, 128)):
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

        # Xavier initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass with automatic tensor conversion.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)
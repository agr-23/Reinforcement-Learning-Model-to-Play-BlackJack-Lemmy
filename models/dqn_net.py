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


    def load_checkpoint_safe(self, checkpoint_path):
        """
        Carga un checkpoint de manera segura en el modelo, ignorando capas incompatibles.
        """
        ckpt = torch.load(checkpoint_path)
        model_dict = self.state_dict()

        # Filtrar solo los pesos que coinciden en nombre y tamaño
        pretrained_dict = {k: v for k, v in ckpt["policy_state"].items()
                           if k in model_dict and v.size() == model_dict[k].size()}

        # Actualizar el modelo con esos pesos
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        print(f"[INFO] Cargadas {len(pretrained_dict)} capas del checkpoint. "
              f"Las demás inicializadas desde cero.")

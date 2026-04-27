import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dims=(512, 512, 512)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.LayerNorm(h), nn.Mish()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)

import math
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1).float() * freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)


class DiffusionMLP(nn.Module):
    def __init__(self, action_dim=28, obs_dim=23, time_dim=128,
                 hidden_dims=(1024, 1024, 1024)):
        super().__init__()
        self.action_dim = action_dim
        self.time_embed = SinusoidalEmbedding(time_dim)

        input_dim = action_dim + obs_dim + time_dim
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.Mish(),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, noisy_action, obs, timestep):
        t_emb = self.time_embed(timestep)
        x = torch.cat([noisy_action, obs, t_emb], dim=-1)
        return self.net(x)

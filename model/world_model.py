import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """Dynamics + reward + done model over action chunks.

    Predicts delta_obs (next_obs = obs + delta) rather than next_obs directly;
    this keeps targets small-magnitude even when obs are normalized to [-1, 1].
    Reward is a scalar for the full chunk; done is a termination logit.
    """

    def __init__(self, obs_dim, action_dim, hidden_dims=(512, 512, 512)):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers = []
        in_dim = obs_dim + action_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.LayerNorm(h), nn.Mish()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.obs_head = nn.Linear(in_dim, obs_dim)
        self.reward_head = nn.Linear(in_dim, 1)
        self.done_head = nn.Linear(in_dim, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        h = self.trunk(x)
        return (
            self.obs_head(h),
            self.reward_head(h).squeeze(-1),
            self.done_head(h).squeeze(-1),
        )

    @torch.no_grad()
    def predict(self, obs, action):
        delta, reward, done_logit = self(obs, action)
        next_obs = obs + delta
        done = torch.sigmoid(done_logit)
        return next_obs, reward, done

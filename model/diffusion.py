import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from model.diffusion_mlp import DiffusionMLP


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim=28, obs_dim=23, n_timesteps=100,
                 hidden_dims=(1024, 1024, 1024), ema_decay=0.995):
        super().__init__()
        self.action_dim = action_dim
        self.n_timesteps = n_timesteps

        self.model = DiffusionMLP(action_dim, obs_dim, hidden_dims=hidden_dims)
        self.ema_model = deepcopy(self.model)
        self.ema_decay = ema_decay

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas_cumprod", torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(torch.tensor(alphas_cumprod, dtype=torch.float32)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - torch.tensor(alphas_cumprod, dtype=torch.float32)))

    @torch.no_grad()
    def update_ema(self):
        for p, p_ema in zip(self.model.parameters(), self.ema_model.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise, noise

    def training_loss(self, action, obs):
        batch_size = action.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=action.device)
        noise = torch.randn_like(action)
        noisy_action, _ = self.q_sample(action, t, noise)
        pred_noise = self.model(noisy_action, obs, t)
        return nn.functional.mse_loss(pred_noise, noise)

    def training_loss_per_sample(self, action, obs):
        batch_size = action.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=action.device)
        noise = torch.randn_like(action)
        noisy_action, _ = self.q_sample(action, t, noise)
        pred_noise = self.model(noisy_action, obs, t)
        return ((pred_noise - noise) ** 2).mean(dim=-1)

    @torch.no_grad()
    def ddim_sample(self, obs, n_steps=10, use_ema=True):
        model = self.ema_model if use_ema else self.model
        device = obs.device
        batch_size = obs.shape[0]

        step_size = self.n_timesteps // n_steps
        seq = list(range(0, self.n_timesteps, step_size))
        seq_next = [-1] + list(seq[:-1])

        x = torch.randn(batch_size, self.action_dim, device=device)
        for i in reversed(range(len(seq))):
            t = seq[i]
            t_next = seq_next[i]

            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = model(x, obs, t_batch)

            alpha_t = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * pred_noise

        return x

    @torch.no_grad()
    def denoise_from_noise(self, w, obs, n_steps=10, use_ema=True):
        """DSRL denoising: start from given noise w instead of random noise."""
        model = self.ema_model if use_ema else self.model
        device = obs.device
        batch_size = obs.shape[0]

        step_size = self.n_timesteps // n_steps
        seq = list(range(0, self.n_timesteps, step_size))
        seq_next = [-1] + list(seq[:-1])

        x = w
        for i in reversed(range(len(seq))):
            t = seq[i]
            t_next = seq_next[i]

            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = model(x, obs, t_batch)

            alpha_t = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * pred_noise

        return x

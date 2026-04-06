import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Box


class DSRLEnvWrapper(gym.Env):
    """Wraps a robosuite env for DSRL training with SB3.

    SAC outputs noise vectors w in [-action_magnitude, action_magnitude].
    The frozen diffusion policy denoises w into environment actions.
    """

    def __init__(self, env, diffusion_policy, obs_normalizer, action_normalizer,
                 obs_keys, action_magnitude=1.5, pred_horizon=4, action_per_step=7,
                 ddim_steps=10, device="cuda"):
        super().__init__()
        self.env = env
        self.dp = diffusion_policy
        self.dp.eval()
        for p in self.dp.parameters():
            p.requires_grad = False

        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer
        self.obs_keys = obs_keys
        self.pred_horizon = pred_horizon
        self.action_per_step = action_per_step
        self.ddim_steps = ddim_steps
        self.device = device

        self.noise_dim = pred_horizon * action_per_step
        self.action_space = Box(
            low=-action_magnitude,
            high=action_magnitude,
            shape=(self.noise_dim,),
            dtype=np.float32,
        )

        obs_dict = self.env.reset()
        obs_sample = self._extract_obs(obs_dict)
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=obs_sample.shape,
            dtype=np.float32,
        )

        self._step_count = 0
        self._current_obs_flat = obs_sample
        self._max_steps = getattr(env, 'horizon', 500)

    def _extract_obs(self, obs_dict):
        env_keys = ["object-state" if k == "object" else k for k in self.obs_keys]
        parts = [np.array(obs_dict[k], dtype=np.float32).flatten() for k in env_keys]
        return np.concatenate(parts)

    def step(self, noise_w):
        obs_norm = self.obs_normalizer.normalize(
            torch.tensor(self._current_obs_flat, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        w = torch.tensor(noise_w, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_norm = self.dp.denoise_from_noise(w, obs_norm, n_steps=self.ddim_steps)
        action_flat = self.action_normalizer.unnormalize(action_norm).squeeze(0).cpu().numpy()
        actions = action_flat.reshape(self.pred_horizon, self.action_per_step)

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for a in actions:
            a = np.clip(a, -1, 1)
            obs_dict, reward, done, info = self.env.step(a)
            self._current_obs_flat = self._extract_obs(obs_dict)
            total_reward += reward
            self._step_count += 1

            if self.env._check_success():
                terminated = True
                info["is_success"] = True
                break
            if done or self._step_count >= self._max_steps:
                truncated = True
                break

        return self._current_obs_flat, total_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs_dict = self.env.reset()
        self._step_count = 0
        self._current_obs_flat = self._extract_obs(obs_dict)
        return self._current_obs_flat.copy(), {}

    def close(self):
        self.env.close()

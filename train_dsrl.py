import argparse
import json
import os
import numpy as np
import torch
import h5py
import robosuite as suite
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from model.diffusion import DiffusionPolicy
from data.normalize import MinMaxNormalizer
from wrappers.dsrl_env import DSRLEnvWrapper


def make_env_from_hdf5(hdf5_path, reward_shaping=False, horizon=700):
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])

    env_kwargs = env_args["env_kwargs"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs["reward_shaping"] = reward_shaping
    env_kwargs["horizon"] = horizon
    env_kwargs.pop("ignore_done", None)

    return suite.make(env_args["env_name"], **env_kwargs)


def collect_initial_rollouts(env, n_env_steps):
    """Roll out BC policy (w ~ N(0,I)) to fill replay buffer with good transitions."""
    obs, _ = env.reset()
    collected = 0
    episodes = 0
    successes = 0

    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []

    while collected < n_env_steps:
        w = np.random.randn(*env.action_space.shape).astype(np.float32)
        w = np.clip(w, env.action_space.low, env.action_space.high)

        observations.append(obs.copy())
        next_obs, reward, terminated, truncated, info = env.step(w)
        next_observations.append(next_obs.copy())
        actions.append(w)
        rewards.append(reward)
        dones.append(terminated or truncated)

        obs = next_obs
        collected += env.pred_horizon

        if terminated or truncated:
            episodes += 1
            if info.get("is_success", False):
                successes += 1
            obs, _ = env.reset()

    print(f"initial rollout: {collected} env steps, {episodes} episodes, "
          f"{successes}/{episodes} successes ({successes/max(episodes,1):.0%})")

    return (np.array(observations), np.array(next_observations),
            np.array(actions), np.array(rewards), np.array(dones))


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, dp, obs_norm, act_norm, obs_keys, save_dir,
                 n_eval=20, eval_freq=10, pred_horizon=4, action_per_step=14,
                 ddim_steps=100, device="cuda"):
        super().__init__()
        self.eval_env = eval_env
        self.dp = dp
        self.obs_norm = obs_norm
        self.act_norm = act_norm
        self.obs_keys = obs_keys
        self.save_dir = save_dir
        self.n_eval = n_eval
        self.eval_freq = eval_freq
        self.pred_horizon = pred_horizon
        self.action_per_step = action_per_step
        self.ddim_steps = ddim_steps
        self.device = device
        self.episode_count = 0
        self.best_sr = 0.0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("TimeLimit.truncated", False) or info.get("is_success", False) or \
               info.get("terminal_observation") is not None:
                self.episode_count += 1
                if self.episode_count % self.eval_freq == 0:
                    sr = self._evaluate()
                    print(f"[eval @ ep {self.episode_count}] success rate: {sr:.1%}")
                    if sr > self.best_sr:
                        self.best_sr = sr
                        path = os.path.join(self.save_dir, "sac_best")
                        self.model.save(path)
                        print(f"  new best! saved to {path}")
        return True

    def _evaluate(self):
        successes = 0
        for _ in range(self.n_eval):
            obs_dict = self.eval_env.reset()
            done = False
            steps = 0
            success = False

            while not done and steps < 700:
                obs_flat = self._extract_obs(obs_dict)
                obs_t = self.obs_norm.normalize(
                    torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0).to(self.device)
                )
                action_norm = self.dp.ddim_sample(obs_t, n_steps=self.ddim_steps)
                action_flat = self.act_norm.unnormalize(action_norm).squeeze(0).cpu().numpy()
                actions = action_flat.reshape(self.pred_horizon, self.action_per_step)

                for a in actions:
                    a = np.clip(a, -1, 1)
                    obs_dict, r, done, info = self.eval_env.step(a)
                    steps += 1
                    if self.eval_env._check_success():
                        success = True
                    if done:
                        break

            successes += int(success)

        return successes / self.n_eval

    def _extract_obs(self, obs_dict):
        env_keys = ["object-state" if k == "object" else k for k in self.obs_keys]
        parts = [np.array(obs_dict[k], dtype=np.float32).flatten() for k in env_keys]
        return np.concatenate(parts)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    obs_normalizer = MinMaxNormalizer.__new__(MinMaxNormalizer)
    obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
    action_normalizer = MinMaxNormalizer.__new__(MinMaxNormalizer)
    action_normalizer.load_state_dict(ckpt["action_normalizer"])

    obs_keys = ckpt["obs_keys"]
    obs_dim = len(obs_normalizer.mins)
    action_dim = len(action_normalizer.mins)
    action_per_step = action_dim // cfg["pred_horizon"]

    policy = DiffusionPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
        n_timesteps=cfg["diffusion_steps"],
        hidden_dims=tuple(cfg["hidden_dims"]),
    ).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    print(f"obs_dim={obs_dim} action_dim={action_dim} action_per_step={action_per_step}")
    print(f"pred_horizon={cfg['pred_horizon']} ddim_steps={args.ddim_steps}")
    print(f"action_magnitude={args.action_magnitude} gamma={args.gamma}")

    train_env_raw = make_env_from_hdf5(cfg["data_path"], reward_shaping=False, horizon=700)
    train_env = DSRLEnvWrapper(
        train_env_raw, policy, obs_normalizer, action_normalizer,
        obs_keys=obs_keys,
        action_magnitude=args.action_magnitude,
        pred_horizon=cfg["pred_horizon"],
        action_per_step=action_per_step,
        ddim_steps=args.ddim_steps,
        device=str(device),
    )

    # collect initial BC rollouts to warm up replay buffer
    print(f"\ncollecting {args.initial_steps} initial env steps with BC policy...")
    init_obs, init_next_obs, init_actions, init_rewards, init_dones = \
        collect_initial_rollouts(train_env, args.initial_steps)

    eval_env = make_env_from_hdf5(cfg["data_path"], reward_shaping=False, horizon=700)

    save_dir = os.path.join("checkpoints/dsrl", args.task)
    os.makedirs(save_dir, exist_ok=True)

    policy_kwargs = dict(
        net_arch=[args.hidden_size] * args.n_layers,
    )

    sac = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        gradient_steps=args.utd,
        train_freq=1,
        target_entropy=args.target_entropy,
        ent_coef=args.ent_coef,
        tau=args.tau,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
    )

    # load initial rollouts into replay buffer
    print(f"loading {len(init_obs)} transitions into replay buffer...")
    for i in range(len(init_obs)):
        sac.replay_buffer.add(
            obs=init_obs[i:i+1],
            next_obs=init_next_obs[i:i+1],
            action=init_actions[i:i+1],
            reward=np.array([init_rewards[i]]),
            done=np.array([init_dones[i]]),
            infos=[{}],
        )

    eval_cb = EvalCallback(
        eval_env, policy, obs_normalizer, action_normalizer,
        obs_keys=obs_keys,
        save_dir=save_dir,
        n_eval=args.n_eval,
        eval_freq=args.eval_freq,
        pred_horizon=cfg["pred_horizon"],
        action_per_step=action_per_step,
        ddim_steps=args.ddim_steps,
        device=str(device),
    )

    total_timesteps = args.train_steps
    print(f"\nstarting SAC training for {total_timesteps} wrapper steps")
    print(f"replay buffer: {sac.replay_buffer.size()} transitions pre-loaded")

    sac.learn(total_timesteps=total_timesteps, callback=eval_cb)

    sac_path = os.path.join(save_dir, "sac_final")
    sac.save(sac_path)
    print(f"saved SAC policy to {sac_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="transport")

    # paper Table 4: Transport hyperparameters
    parser.add_argument("--action_magnitude", type=float, default=1.0)
    parser.add_argument("--ddim_steps", type=int, default=10)
    parser.add_argument("--initial_steps", type=int, default=320000)
    parser.add_argument("--train_steps", type=int, default=100000)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--utd", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target_entropy", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=1000000)

    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--n_eval", type=int, default=20)
    parser.add_argument("--eval_freq", type=int, default=20)
    args = parser.parse_args()
    train(args)

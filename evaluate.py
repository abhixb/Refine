import argparse
import json
import numpy as np
import torch
import h5py
import robosuite as suite

from model.diffusion import DiffusionPolicy
from data.normalize import MinMaxNormalizer

def hdf5_to_env_key(k):
    return "object-state" if k == "object" else k


def make_env_from_hdf5(hdf5_path, reward_shaping=False, render=False):
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])

    env_kwargs = env_args["env_kwargs"]
    env_kwargs["has_renderer"] = render
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs["reward_shaping"] = reward_shaping
    env_kwargs["horizon"] = 700
    env_kwargs.pop("ignore_done", None)

    env = suite.make(env_args["env_name"], **env_kwargs)
    return env


def extract_obs(obs_dict, obs_keys):
    env_keys = [hdf5_to_env_key(k) for k in obs_keys]
    parts = [np.array(obs_dict[k], dtype=np.float32).flatten() for k in env_keys]
    return np.concatenate(parts)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    obs_normalizer = MinMaxNormalizer.__new__(MinMaxNormalizer)
    obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
    action_normalizer = MinMaxNormalizer.__new__(MinMaxNormalizer)
    action_normalizer.load_state_dict(ckpt["action_normalizer"])

    obs_dim = len(obs_normalizer.mins)
    action_dim = len(action_normalizer.mins)

    policy = DiffusionPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
        n_timesteps=cfg["diffusion_steps"],
        hidden_dims=tuple(cfg["hidden_dims"]),
    ).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    obs_keys = ckpt.get("obs_keys", ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"])
    action_per_step = action_dim // cfg["pred_horizon"]

    env = make_env_from_hdf5(cfg["data_path"], reward_shaping=False, render=args.render)

    successes = 0
    for ep in range(args.n_episodes):
        obs_dict = env.reset()
        done = False
        step_count = 0
        ep_reward = 0.0
        ep_success = False

        while not done and step_count < 700:
            obs_flat = extract_obs(obs_dict, obs_keys)
            obs_norm = obs_normalizer.normalize(
                torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0).to(device)
            )

            action_norm = policy.ddim_sample(obs_norm, n_steps=args.ddim_steps)
            action_flat = action_normalizer.unnormalize(action_norm).squeeze(0).cpu().numpy()
            actions = action_flat.reshape(cfg["pred_horizon"], action_per_step)

            for a in actions:
                a = np.clip(a, -1, 1)
                obs_dict, reward, done, info = env.step(a)
                ep_reward += reward
                step_count += 1
                if env._check_success():
                    ep_success = True
                if done:
                    break

        successes += int(ep_success)
        print(f"episode {ep + 1:03d} | reward {ep_reward:.2f} | success {ep_success}")

    rate = successes / args.n_episodes
    print(f"\nsuccess rate: {successes}/{args.n_episodes} = {rate:.1%}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="lift")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--ddim_steps", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(args)

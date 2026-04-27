import argparse
import json
import os
import numpy as np
import torch
import h5py
import imageio
import robosuite as suite

from model.diffusion import DiffusionPolicy
from data.normalize import MinMaxNormalizer


def make_env_from_hdf5(hdf5_path, horizon=700):
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])

    env_kwargs = env_args["env_kwargs"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_names"] = ["agentview"]
    env_kwargs["camera_heights"] = 512
    env_kwargs["camera_widths"] = 512
    env_kwargs["reward_shaping"] = False
    env_kwargs["horizon"] = horizon
    env_kwargs.pop("ignore_done", None)
    env_kwargs.pop("camera_depths", None)

    return suite.make(env_args["env_name"], **env_kwargs)


def extract_obs(obs_dict, obs_keys):
    env_keys = ["object-state" if k == "object" else k for k in obs_keys]
    parts = [np.array(obs_dict[k], dtype=np.float32).flatten() for k in env_keys]
    return np.concatenate(parts)


def capture_home_pose(obs_dict, n_arms):
    return [np.array(obs_dict[f"robot{i}_eef_pos"], dtype=np.float32).copy() for i in range(n_arms)]


def home_action(obs_dict, home_pos, n_arms, k_pos=10.0):
    parts = []
    for i in range(n_arms):
        cur = np.array(obs_dict[f"robot{i}_eef_pos"], dtype=np.float32)
        d_pos = np.clip(k_pos * (home_pos[i] - cur), -1.0, 1.0)
        parts.append(np.concatenate([d_pos, np.zeros(3, dtype=np.float32), [-1.0]]))
    return np.concatenate(parts)


def at_home(obs_dict, home_pos, n_arms, tol=0.03):
    for i in range(n_arms):
        cur = np.array(obs_dict[f"robot{i}_eef_pos"], dtype=np.float32)
        if np.linalg.norm(cur - home_pos[i]) > tol:
            return False
    return True


def record(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    obs_keys = ckpt["obs_keys"]

    obs_normalizer = MinMaxNormalizer.__new__(MinMaxNormalizer)
    obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
    action_normalizer = MinMaxNormalizer.__new__(MinMaxNormalizer)
    action_normalizer.load_state_dict(ckpt["action_normalizer"])

    obs_dim = len(obs_normalizer.mins)
    action_dim = len(action_normalizer.mins)
    pred_horizon = cfg["pred_horizon"]
    action_per_step = action_dim // pred_horizon
    exec_horizon = args.exec_horizon if args.exec_horizon is not None else pred_horizon
    exec_horizon = max(1, min(exec_horizon, pred_horizon))

    policy = DiffusionPolicy(
        action_dim=action_dim, obs_dim=obs_dim,
        n_timesteps=cfg["diffusion_steps"],
        hidden_dims=tuple(cfg["hidden_dims"]),
    ).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    env = make_env_from_hdf5(cfg["data_path"], horizon=args.horizon)
    os.makedirs(args.out_dir, exist_ok=True)

    n_arms = action_per_step // 7

    for ep in range(args.n_episodes):
        obs_dict = env.reset()
        home_pos = capture_home_pose(obs_dict, n_arms)
        frames = []
        done = False
        step_count = 0
        ep_success = False
        home_steps = 0

        while not done and step_count < args.horizon:
            frames.append(obs_dict["agentview_image"][::-1])

            if ep_success:
                if at_home(obs_dict, home_pos, n_arms) or home_steps >= args.home_max_steps:
                    break
                a = home_action(obs_dict, home_pos, n_arms)
                a = np.clip(a, -1, 1)
                obs_dict, reward, done, info = env.step(a)
                step_count += 1
                home_steps += 1
                continue

            obs_flat = extract_obs(obs_dict, obs_keys)
            obs_norm = obs_normalizer.normalize(
                torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0).to(device)
            )

            action_norm = policy.ddim_sample(obs_norm, n_steps=10)

            action_flat = action_normalizer.unnormalize(action_norm).squeeze(0).cpu().numpy()
            actions = action_flat.reshape(pred_horizon, action_per_step)

            for a in actions[:exec_horizon]:
                a = np.clip(a, -1, 1)
                obs_dict, reward, done, info = env.step(a)
                step_count += 1
                if env._check_success():
                    ep_success = True
                    break
                if done:
                    break

        tag = "success" if ep_success else "fail"
        path = os.path.join(args.out_dir, f"{args.task}_ep{ep + 1:02d}_{tag}.mp4")
        imageio.mimsave(path, frames, fps=20)
        print(f"saved {path} ({len(frames)} frames, {tag})")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--exec_horizon", type=int, default=None,
                        help="actions to execute per chunk before resampling. "
                             "Defaults to pred_horizon. Lower for closed-loop.")
    parser.add_argument("--home_max_steps", type=int, default=80,
                        help="step cap for the post-success go-home controller")
    parser.add_argument("--out_dir", type=str, default="videos")
    args = parser.parse_args()
    record(args)

import argparse
import json
import os
import numpy as np
import torch
import h5py
import imageio
import robosuite as suite
from stable_baselines3 import SAC

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
    action_per_step = action_dim // cfg["pred_horizon"]

    policy = DiffusionPolicy(
        action_dim=action_dim, obs_dim=obs_dim,
        n_timesteps=cfg["diffusion_steps"],
        hidden_dims=tuple(cfg["hidden_dims"]),
    ).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    sac_model = None
    if args.sac_checkpoint:
        sac_model = SAC.load(args.sac_checkpoint, device=device)
        print(f"loaded SAC from {args.sac_checkpoint}")

    env = make_env_from_hdf5(cfg["data_path"], horizon=args.horizon)
    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(args.n_episodes):
        obs_dict = env.reset()
        frames = []
        done = False
        step_count = 0
        ep_success = False

        while not done and step_count < args.horizon:
            frames.append(obs_dict["agentview_image"][::-1])

            obs_flat = extract_obs(obs_dict, obs_keys)
            obs_norm = obs_normalizer.normalize(
                torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0).to(device)
            )

            if sac_model is not None:
                w, _ = sac_model.predict(obs_flat, deterministic=True)
                w_t = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device)
                action_norm = policy.denoise_from_noise(w_t, obs_norm, n_steps=10)
            else:
                action_norm = policy.ddim_sample(obs_norm, n_steps=10)

            action_flat = action_normalizer.unnormalize(action_norm).squeeze(0).cpu().numpy()
            actions = action_flat.reshape(cfg["pred_horizon"], action_per_step)

            for a in actions:
                a = np.clip(a, -1, 1)
                obs_dict, reward, done, info = env.step(a)
                step_count += 1
                if env._check_success():
                    ep_success = True
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
    parser.add_argument("--sac_checkpoint", type=str, default=None)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--out_dir", type=str, default="videos")
    args = parser.parse_args()
    record(args)

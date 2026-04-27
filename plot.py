import argparse
import json
import os
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import robosuite as suite

from model.diffusion import DiffusionPolicy
from data.normalize import MinMaxNormalizer


def make_env(hdf5_path, horizon=700):
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])

    kw = env_args["env_kwargs"]
    kw["has_renderer"] = False
    kw["has_offscreen_renderer"] = False
    kw["use_camera_obs"] = False
    kw["reward_shaping"] = False
    kw["horizon"] = horizon
    for k in ("ignore_done", "camera_depths", "camera_names", "camera_heights", "camera_widths"):
        kw.pop(k, None)
    return suite.make(env_args["env_name"], **kw)


def extract_obs(obs_dict, obs_keys):
    env_keys = ["object-state" if k == "object" else k for k in obs_keys]
    parts = [np.array(obs_dict[k], dtype=np.float32).flatten() for k in env_keys]
    return np.concatenate(parts)


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


def rollout(args):
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
    n_arms = action_per_step // 7
    exec_horizon = max(1, min(args.exec_horizon, pred_horizon))

    policy = DiffusionPolicy(
        action_dim=action_dim, obs_dim=obs_dim,
        n_timesteps=cfg["diffusion_steps"],
        hidden_dims=tuple(cfg["hidden_dims"]),
    ).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    env = make_env(cfg["data_path"], horizon=args.horizon)
    obs_dict = env.reset()
    home_pos = [np.array(obs_dict[f"robot{i}_eef_pos"], dtype=np.float32).copy() for i in range(n_arms)]

    eef_traj = []
    phases = []
    ep_success = False
    success_step = None
    home_steps = 0
    step_count = 0
    done = False

    def record_step(phase):
        eef_traj.append(np.array(obs_dict["robot0_eef_pos"], dtype=np.float32).copy())
        phases.append(phase)

    while not done and step_count < args.horizon:
        if ep_success:
            record_step("go_home")
            if at_home(obs_dict, home_pos, n_arms) or home_steps >= args.home_max_steps:
                break
            a = np.clip(home_action(obs_dict, home_pos, n_arms), -1, 1)
            obs_dict, _, done, _ = env.step(a)
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
            record_step("policy")
            a = np.clip(a, -1, 1)
            obs_dict, _, done, _ = env.step(a)
            step_count += 1
            if env._check_success() and not ep_success:
                ep_success = True
                success_step = len(eef_traj)
                break
            if done:
                break

    env.close()
    return np.stack(eef_traj), np.array(phases), home_pos[0], success_step, ep_success


def plot_rollout(traj, phases, home, success_step, task, out_path):
    is_home = phases == "go_home"
    is_pol = ~is_home
    t = np.arange(len(traj))
    dist = np.linalg.norm(traj - home, axis=1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(12, 4.8))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    if is_pol.any():
        p = traj[is_pol]
        ax1.plot(p[:, 0], p[:, 1], p[:, 2], color="#1f77b4", lw=2.0, label="policy")
    if is_home.any():
        p = traj[is_home]
        ax1.plot(p[:, 0], p[:, 1], p[:, 2], color="#d62728", lw=2.0, label="go-home")
    ax1.scatter(*home, color="#2ca02c", s=120, marker="*", edgecolor="k",
                linewidth=0.6, label="home", zorder=10)
    if success_step is not None and success_step < len(traj):
        ax1.scatter(*traj[success_step], color="#ff7f0e", s=90, marker="o",
                    edgecolor="k", linewidth=0.6, label="success", zorder=10)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.set_title(f"{task} — eef trajectory")
    ax1.legend(loc="upper left", fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, dist, color="#222", lw=1.8)
    ymax = dist.max() * 1.05
    ax2.fill_between(t, 0, ymax, where=is_home, color="#d62728", alpha=0.18,
                     label="go-home phase", step="post")
    if success_step is not None:
        ax2.axvline(success_step, color="#ff7f0e", ls="--", lw=1.4,
                    label=f"success @ t={success_step}")
    ax2.set_xlabel("env step")
    ax2.set_ylabel(r"$\|eef - home\|$  (m)")
    ax2.set_title("distance to home")
    ax2.set_ylim(0, ymax)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--exec_horizon", type=int, default=2)
    parser.add_argument("--home_max_steps", type=int, default=80)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = f"figures/{args.task}_rollout.png"

    traj, phases, home, success_step, success = rollout(args)
    print(f"steps={len(traj)} success={success} success_step={success_step}")
    plot_rollout(traj, phases, home, success_step, args.task, args.out)


if __name__ == "__main__":
    main()

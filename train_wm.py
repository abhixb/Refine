import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import yaml

from data.normalize import MinMaxNormalizer
from model.diffusion import DiffusionPolicy
from model.world_model import WorldModel
from train_awr import collect_rollouts_parallel
from wrappers.parallel_envs import ParallelEnvs


def flatten_episodes(episodes):
    obs, actions, next_obs, rewards, dones = [], [], [], [], []
    for ep in episodes:
        obs.append(np.stack(ep["obs"]))
        actions.append(np.stack(ep["actions"]))
        next_obs.append(np.stack(ep["next_obs"]))
        rewards.append(np.asarray(ep["rewards"], dtype=np.float32))
        dones.append(np.asarray(ep["dones"], dtype=np.float32))
    return (
        np.concatenate(obs, axis=0),
        np.concatenate(actions, axis=0),
        np.concatenate(next_obs, axis=0),
        np.concatenate(rewards, axis=0),
        np.concatenate(dones, axis=0),
    )


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cfg["bc_checkpoint"], map_location=device, weights_only=False)
    bc_cfg = ckpt["config"]

    obs_normalizer = MinMaxNormalizer.__new__(MinMaxNormalizer)
    obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
    action_normalizer = MinMaxNormalizer.__new__(MinMaxNormalizer)
    action_normalizer.load_state_dict(ckpt["action_normalizer"])

    obs_dim = len(obs_normalizer.mins)
    action_dim = len(action_normalizer.mins)
    pred_horizon = bc_cfg["pred_horizon"]
    action_per_step = action_dim // pred_horizon
    obs_keys = ckpt["obs_keys"]

    policy = DiffusionPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
        n_timesteps=bc_cfg["diffusion_steps"],
        hidden_dims=tuple(bc_cfg["hidden_dims"]),
    ).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    os.makedirs(cfg["save_dir"], exist_ok=True)

    parallel = ParallelEnvs(
        n_envs=cfg["n_envs"],
        hdf5_path=cfg["demo_data_path"],
        horizon=cfg["horizon"],
        obs_keys=obs_keys,
        task=cfg["task"],
        pred_horizon=pred_horizon,
        action_per_step=action_per_step,
    )

    print(f"collecting {cfg['rollout_episodes']} rollouts with BC policy...")
    t0 = time.time()
    episodes, succ, mean_ret = collect_rollouts_parallel(
        parallel, policy, obs_normalizer, action_normalizer,
        pred_horizon, action_per_step, cfg["ddim_steps"],
        cfg["rollout_episodes"], device,
    )
    parallel.close()
    print(f"rollouts done in {time.time() - t0:.1f}s | success {succ:.2%} | mean return {mean_ret:.2f}")

    obs, actions, next_obs, rewards, dones = flatten_episodes(episodes)
    print(f"dataset: {obs.shape[0]} chunk transitions")

    delta = next_obs - obs
    delta_mean = delta.mean(axis=0)
    delta_std = delta.std(axis=0) + 1e-6
    r_mean = float(rewards.mean())
    r_std = float(rewards.std() + 1e-6)
    print(f"delta_obs std range [{delta_std.min():.4f}, {delta_std.max():.4f}] | reward mean {r_mean:.3f} std {r_std:.3f}")

    obs_t = torch.tensor(obs, dtype=torch.float32)
    act_t = torch.tensor(actions, dtype=torch.float32)
    d_obs_t = torch.tensor((delta - delta_mean) / delta_std, dtype=torch.float32)
    r_t = torch.tensor((rewards - r_mean) / r_std, dtype=torch.float32)
    done_t = torch.tensor(dones, dtype=torch.float32)

    wm = WorldModel(obs_dim=obs_dim, action_dim=action_dim,
                    hidden_dims=tuple(cfg["hidden_dims"])).to(device)
    opt = torch.optim.AdamW(wm.parameters(), lr=cfg["lr"], weight_decay=1e-6)

    n = obs.shape[0]
    batch = cfg["batch_size"]
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["done_pos_weight"], device=device))

    for step in range(1, cfg["steps"] + 1):
        idx = np.random.randint(0, n, size=batch)
        b_obs = obs_t[idx].to(device)
        b_act = act_t[idx].to(device)
        b_dobs = d_obs_t[idx].to(device)
        b_r = r_t[idx].to(device)
        b_done = done_t[idx].to(device)

        delta_pred, r_pred, done_logit = wm(b_obs, b_act)
        loss_obs = nn.functional.mse_loss(delta_pred, b_dobs)
        loss_r = nn.functional.mse_loss(r_pred, b_r)
        loss_done = bce(done_logit, b_done)
        loss = loss_obs + cfg["reward_weight"] * loss_r + cfg["done_weight"] * loss_done

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()

        if step % cfg["log_every"] == 0:
            print(f"step {step:05d} | loss {loss.item():.4f} "
                  f"| obs {loss_obs.item():.4f} | r {loss_r.item():.4f} | done {loss_done.item():.4f}")

    torch.save({
        "wm": wm.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dims": cfg["hidden_dims"],
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "reward_mean": r_mean,
        "reward_std": r_std,
        "bc_checkpoint": cfg["bc_checkpoint"],
    }, os.path.join(cfg["save_dir"], "wm.pt"))
    print(f"saved {cfg['save_dir']}/wm.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/wm_transport.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)

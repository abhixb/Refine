import argparse
import csv
import json
import os
import time
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import robosuite as suite

from data.dataset import load_robomimic_hdf5, detect_obs_keys
from data.normalize import MinMaxNormalizer
from model.diffusion import DiffusionPolicy
from model.critic import ValueNet
from wrappers.transport_reward import transport_dense_reward
from wrappers.parallel_envs import ParallelEnvs


DENSE_REWARDS = {"transport": transport_dense_reward}


def hdf5_to_env_key(k):
    return "object-state" if k == "object" else k


def make_env_from_hdf5(hdf5_path, horizon=700):
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])

    env_kwargs = env_args["env_kwargs"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs["reward_shaping"] = False
    env_kwargs["horizon"] = horizon
    env_kwargs.pop("ignore_done", None)

    return suite.make(env_args["env_name"], **env_kwargs)


def extract_obs(obs_dict, obs_keys):
    env_keys = [hdf5_to_env_key(k) for k in obs_keys]
    parts = [np.array(obs_dict[k], dtype=np.float32).flatten() for k in env_keys]
    return np.concatenate(parts)


def collect_rollouts_parallel(parallel, policy, obs_normalizer, action_normalizer,
                              pred_horizon, action_per_step, ddim_steps,
                              target_episodes, device):
    N = parallel.n
    completed = []
    successes = 0
    total_return = 0.0

    while len(completed) < target_episodes:
        current_obs = parallel.reset_all()
        eps = [{"obs": [], "actions": [], "rewards": [], "next_obs": [], "dones": []}
               for _ in range(N)]
        ep_success = [False] * N
        active = [True] * N

        while any(active):
            active_idx = [i for i in range(N) if active[i]]
            obs_batch = torch.tensor(
                np.stack([current_obs[i] for i in active_idx]),
                dtype=torch.float32, device=device,
            )
            obs_norm = obs_normalizer.normalize(obs_batch)
            action_norm = policy.ddim_sample(obs_norm, n_steps=ddim_steps, use_ema=False)
            actions = action_normalizer.unnormalize(action_norm).cpu().numpy()
            obs_norm_np = obs_norm.cpu().numpy()
            action_norm_np = action_norm.cpu().numpy()

            for local, i in enumerate(active_idx):
                parallel.step_send(i, actions[local])

            for local, i in enumerate(active_idx):
                next_obs_flat, chunk_reward, chunk_done, chunk_success = parallel.step_recv(i)
                next_obs_norm = obs_normalizer.normalize(
                    torch.tensor(next_obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
                ).squeeze(0).cpu().numpy()

                eps[i]["obs"].append(obs_norm_np[local])
                eps[i]["actions"].append(action_norm_np[local])
                eps[i]["rewards"].append(float(chunk_reward))
                eps[i]["next_obs"].append(next_obs_norm)
                eps[i]["dones"].append(bool(chunk_done))

                if chunk_success:
                    ep_success[i] = True

                if chunk_done:
                    active[i] = False
                else:
                    current_obs[i] = next_obs_flat

        for i in range(N):
            if len(completed) >= target_episodes:
                break
            completed.append(eps[i])
            successes += int(ep_success[i])
            total_return += float(sum(eps[i]["rewards"]))

    n = len(completed)
    return completed, successes / max(n, 1), total_return / max(n, 1)


def compute_gae(episodes, critic, gamma, gae_lambda, device):
    all_obs, all_actions, all_returns, all_advantages = [], [], [], []

    critic.eval()
    with torch.no_grad():
        for ep in episodes:
            obs = torch.tensor(np.stack(ep["obs"]), dtype=torch.float32, device=device)
            next_obs = torch.tensor(np.stack(ep["next_obs"]), dtype=torch.float32, device=device)
            rewards = np.array(ep["rewards"], dtype=np.float32)
            dones = np.array(ep["dones"], dtype=np.float32)

            v = critic(obs).cpu().numpy()
            v_next = critic(next_obs).cpu().numpy()

            T = len(rewards)
            adv = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                delta = rewards[t] + gamma * v_next[t] * (1.0 - dones[t]) - v[t]
                gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
                adv[t] = gae
            returns = adv + v

            all_obs.append(np.stack(ep["obs"]))
            all_actions.append(np.stack(ep["actions"]))
            all_returns.append(returns)
            all_advantages.append(adv)

    obs = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    returns = np.concatenate(all_returns, axis=0)
    advantages = np.concatenate(all_advantages, axis=0)
    return obs, actions, returns, advantages


def update_critic(critic, optimizer, obs, returns, steps, batch_size, device):
    critic.train()
    n = obs.shape[0]
    losses = []
    for _ in range(steps):
        idx = np.random.randint(0, n, size=batch_size)
        b_obs = torch.tensor(obs[idx], dtype=torch.float32, device=device)
        b_ret = torch.tensor(returns[idx], dtype=torch.float32, device=device)
        pred = critic(b_obs)
        loss = nn.functional.mse_loss(pred, b_ret)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def update_policy(policy, optimizer, rollout_obs, rollout_actions, rollout_adv,
                  demo_obs, demo_actions, steps, batch_size, demo_frac,
                  beta, weight_clip, adv_norm, positive_adv_only, device):
    policy.model.train()

    n_demo_target = int(round(batch_size * demo_frac))
    n_rollout_target = batch_size - n_demo_target

    if adv_norm == "std":
        adv = rollout_adv - rollout_adv.mean()
        adv = adv / (rollout_adv.std() + 1e-8)
    elif adv_norm == "mean":
        adv = rollout_adv - rollout_adv.mean()
    else:
        adv = rollout_adv.copy()

    if positive_adv_only:
        pool = np.where(adv > 0)[0]
    else:
        pool = np.arange(adv.shape[0])

    n_demo_total = demo_obs.shape[0]
    pool_size = pool.shape[0]
    frac_positive = pool_size / max(adv.shape[0], 1)

    losses = []
    mean_w = []
    for _ in range(steps):
        if pool_size > 0:
            ridx = pool[np.random.randint(0, pool_size, size=n_rollout_target)]
            n_rollout = n_rollout_target
            n_demo = n_demo_target
        else:
            ridx = None
            n_rollout = 0
            n_demo = batch_size

        didx = np.random.randint(0, n_demo_total, size=n_demo)
        d_obs = demo_obs[didx].to(device)
        d_act = demo_actions[didx].to(device)
        w_demo = torch.ones(n_demo, device=device)

        if n_rollout > 0:
            r_obs = torch.tensor(rollout_obs[ridx], dtype=torch.float32, device=device)
            r_act = torch.tensor(rollout_actions[ridx], dtype=torch.float32, device=device)
            r_adv = torch.tensor(adv[ridx], dtype=torch.float32, device=device)
            w_roll = torch.clamp(torch.exp(r_adv / beta), max=weight_clip)

            obs_batch = torch.cat([r_obs, d_obs], dim=0)
            act_batch = torch.cat([r_act, d_act], dim=0)
            w_batch = torch.cat([w_roll, w_demo], dim=0)
        else:
            obs_batch = d_obs
            act_batch = d_act
            w_batch = w_demo
            w_roll = torch.zeros(1, device=device)

        per_sample = policy.training_loss_per_sample(act_batch, obs_batch)
        loss = (w_batch * per_sample).sum() / (w_batch.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
        optimizer.step()
        policy.update_ema()

        losses.append(loss.item())
        mean_w.append(w_roll.mean().item())

    return float(np.mean(losses)), float(np.mean(mean_w)), frac_positive


def evaluate_policy(env, policy, obs_normalizer, action_normalizer, obs_keys,
                    pred_horizon, action_per_step, ddim_steps, n_episodes,
                    horizon, device):
    successes = 0
    for _ in range(n_episodes):
        obs_dict = env.reset()
        obs_flat = extract_obs(obs_dict, obs_keys)
        ep_success = False
        step_count = 0
        done = False
        while not done and step_count < horizon:
            obs_norm_t = obs_normalizer.normalize(
                torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0).to(device)
            )
            action_norm = policy.ddim_sample(obs_norm_t, n_steps=ddim_steps, use_ema=True)
            action_flat = action_normalizer.unnormalize(action_norm).squeeze(0).cpu().numpy()
            actions = action_flat.reshape(pred_horizon, action_per_step)
            for a in actions:
                a = np.clip(a, -1, 1)
                obs_dict, _, env_done, _ = env.step(a)
                step_count += 1
                if env._check_success():
                    ep_success = True
                    done = True
                    break
                if env_done or step_count >= horizon:
                    done = True
                    break
            obs_flat = extract_obs(obs_dict, obs_keys)
        successes += int(ep_success)
    return successes / max(n_episodes, 1)


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

    obs_keys = ckpt.get("obs_keys", detect_obs_keys(cfg["demo_data_path"]))

    policy = DiffusionPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
        n_timesteps=bc_cfg["diffusion_steps"],
        hidden_dims=tuple(bc_cfg["hidden_dims"]),
        ema_decay=cfg.get("ema_decay", 0.995),
    ).to(device)
    policy.load_state_dict(ckpt["model"])

    critic = ValueNet(obs_dim=obs_dim, hidden_dims=tuple(cfg["critic_hidden"])).to(device)

    policy_opt = torch.optim.AdamW(policy.model.parameters(), lr=cfg["policy_lr"])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg["critic_lr"])

    obs_raw, act_raw, _ = load_robomimic_hdf5(
        cfg["demo_data_path"], pred_horizon=pred_horizon, obs_keys=obs_keys
    )
    demo_obs = torch.tensor(obs_normalizer.normalize(obs_raw), dtype=torch.float32)
    demo_actions = torch.tensor(action_normalizer.normalize(act_raw), dtype=torch.float32)
    print(f"demos loaded: {demo_obs.shape[0]} samples | obs {obs_dim} | action {action_dim}")

    task_name = cfg.get("task", "transport")
    if task_name not in DENSE_REWARDS:
        raise ValueError(f"no dense reward registered for task '{task_name}'")

    eval_env = make_env_from_hdf5(cfg["demo_data_path"], horizon=cfg["horizon"])
    parallel = ParallelEnvs(
        n_envs=cfg["n_envs"],
        hdf5_path=cfg["demo_data_path"],
        horizon=cfg["horizon"],
        obs_keys=obs_keys,
        task=task_name,
        pred_horizon=pred_horizon,
        action_per_step=action_per_step,
    )

    os.makedirs(cfg["save_dir"], exist_ok=True)
    best_success = 0.0

    metrics = {"iter": [], "roll_success": [], "roll_return": [],
               "critic_loss": [], "policy_loss": [], "mean_weight": [],
               "eval_iter": [], "eval_success": []}
    csv_path = os.path.join(cfg["save_dir"], "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "iter", "roll_success", "roll_return",
            "critic_loss", "policy_loss", "mean_weight", "eval_success",
        ])

    for it in range(1, cfg["iterations"] + 1):
        t_start = time.time()
        episodes, roll_success, roll_return = collect_rollouts_parallel(
            parallel, policy, obs_normalizer, action_normalizer,
            pred_horizon, action_per_step, cfg["ddim_steps"],
            cfg["rollouts_per_iter"], device,
        )
        t_roll = time.time() - t_start

        obs, actions, returns, advantages = compute_gae(
            episodes, critic, cfg["gamma"], cfg["gae_lambda"], device
        )

        t_update = time.time()
        critic_loss = update_critic(
            critic, critic_opt, obs, returns,
            cfg["critic_steps"], cfg["critic_batch_size"], device,
        )

        warmup = it <= cfg.get("critic_warmup_iters", 0)
        if warmup:
            policy_loss, mean_weight, frac_pos = 0.0, 0.0, 0.0
            tag = "warmup"
        else:
            policy_loss, mean_weight, frac_pos = update_policy(
                policy, policy_opt, obs, actions, advantages,
                demo_obs, demo_actions,
                cfg["policy_steps"], cfg["policy_batch_size"], cfg["demo_frac"],
                cfg["beta"], cfg["weight_clip"], cfg["advantage_norm"],
                cfg.get("positive_adv_only", True), device,
            )
            tag = f"p_loss {policy_loss:.4f} | mean_w {mean_weight:.2f} | pos {frac_pos:.2f}"

        t_total = time.time() - t_start
        print(
            f"iter {it:04d} | {t_total:.1f}s (roll {t_roll:.1f}s, upd {time.time()-t_update:.1f}s) "
            f"| roll_succ {roll_success:.2%} | roll_ret {roll_return:.2f} "
            f"| v_loss {critic_loss:.4f} | {tag}"
        )

        metrics["iter"].append(it)
        metrics["roll_success"].append(roll_success)
        metrics["roll_return"].append(roll_return)
        metrics["critic_loss"].append(critic_loss)
        metrics["policy_loss"].append(policy_loss)
        metrics["mean_weight"].append(mean_weight)

        eval_success_row = ""
        if it % cfg["eval_every"] == 0 or it == cfg["iterations"]:
            eval_success = evaluate_policy(
                eval_env, policy, obs_normalizer, action_normalizer, obs_keys,
                pred_horizon, action_per_step, cfg["ddim_steps"],
                cfg["eval_episodes"], cfg["horizon"], device,
            )
            print(f"eval iter {it:04d} | success {eval_success:.2%}")
            metrics["eval_iter"].append(it)
            metrics["eval_success"].append(eval_success)
            eval_success_row = f"{eval_success:.6f}"

            if eval_success > best_success:
                best_success = eval_success
                save_checkpoint(policy, critic, obs_normalizer, action_normalizer,
                                obs_keys, bc_cfg, cfg, it,
                                os.path.join(cfg["save_dir"], "awr_best.pt"))

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                it, roll_success, roll_return,
                critic_loss, policy_loss, mean_weight, eval_success_row,
            ])
        plot_metrics(metrics, os.path.join(cfg["save_dir"], "training.png"))

        if it % cfg["save_every"] == 0 or it == cfg["iterations"]:
            save_checkpoint(policy, critic, obs_normalizer, action_normalizer,
                            obs_keys, bc_cfg, cfg, it,
                            os.path.join(cfg["save_dir"], f"state_{it}.pt"))

    parallel.close()
    eval_env.close()
    print(f"done. best eval success: {best_success:.2%}")


def plot_metrics(m, path):
    if len(m["iter"]) == 0:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(m["iter"], m["policy_loss"], color="tab:blue")
    axes[0, 0].set_title("policy loss (weighted)")
    axes[0, 0].set_xlabel("iter")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(m["iter"], m["critic_loss"], color="tab:orange")
    axes[0, 1].set_title("critic MSE")
    axes[0, 1].set_xlabel("iter")
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(m["iter"], m["mean_weight"], color="tab:green")
    axes[0, 2].set_title("mean AWR weight (rollouts)")
    axes[0, 2].set_xlabel("iter")
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(m["iter"], m["roll_success"], color="tab:red", label="rollout")
    if len(m["eval_iter"]):
        axes[1, 0].plot(m["eval_iter"], m["eval_success"], "o-", color="tab:purple", label="eval")
    axes[1, 0].set_title("success rate")
    axes[1, 0].set_xlabel("iter")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(m["iter"], m["roll_return"], color="tab:brown")
    axes[1, 1].set_title("mean rollout return (shaped)")
    axes[1, 1].set_xlabel("iter")
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_checkpoint(policy, critic, obs_normalizer, action_normalizer,
                    obs_keys, bc_cfg, cfg, iteration, path):
    torch.save({
        "iteration": iteration,
        "model": policy.state_dict(),
        "critic": critic.state_dict(),
        "obs_normalizer": obs_normalizer.state_dict(),
        "action_normalizer": action_normalizer.state_dict(),
        "obs_keys": obs_keys,
        "config": {**bc_cfg, **cfg},
    }, path)
    print(f"saved {path}")


def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/awr_transport.yaml")
    parser.add_argument("--bc_checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--rollouts_per_iter", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, {
        "bc_checkpoint": args.bc_checkpoint,
        "save_dir": args.save_dir,
        "iterations": args.iterations,
        "rollouts_per_iter": args.rollouts_per_iter,
        "beta": args.beta,
    })
    train(cfg)

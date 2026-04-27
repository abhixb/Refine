import argparse
import time
import numpy as np
import torch

from data.normalize import MinMaxNormalizer
from model.diffusion import DiffusionPolicy
from model.world_model import WorldModel
from train_awr import make_env_from_hdf5, extract_obs


def load_bc(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    obs_norm = MinMaxNormalizer.__new__(MinMaxNormalizer)
    obs_norm.load_state_dict(ckpt["obs_normalizer"])
    action_norm = MinMaxNormalizer.__new__(MinMaxNormalizer)
    action_norm.load_state_dict(ckpt["action_normalizer"])
    obs_dim = len(obs_norm.mins)
    action_dim = len(action_norm.mins)
    policy = DiffusionPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
        n_timesteps=cfg["diffusion_steps"],
        hidden_dims=tuple(cfg["hidden_dims"]),
    ).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    return policy, obs_norm, action_norm, ckpt["obs_keys"], cfg


def load_wm(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    wm = WorldModel(obs_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
                    hidden_dims=tuple(ckpt["hidden_dims"])).to(device)
    wm.load_state_dict(ckpt["wm"])
    wm.eval()
    stats = {
        "delta_mean": torch.tensor(ckpt["delta_mean"], dtype=torch.float32, device=device),
        "delta_std": torch.tensor(ckpt["delta_std"], dtype=torch.float32, device=device),
        "reward_mean": float(ckpt["reward_mean"]),
        "reward_std": float(ckpt["reward_std"]),
    }
    return wm, stats


def plan_mpc(obs_norm, policy, wm, stats, n_samples, horizon, gamma,
             ddim_steps, device):
    """Closed-loop MPC step: return the best action chunk (normalized).

    Expands obs to N parallel imagined rollouts of length `horizon` chunks,
    each step re-sampling from the diffusion policy at the imagined state.
    Returns the first-step action chunk of the best-scoring trajectory.
    """
    state = obs_norm.expand(n_samples, -1).contiguous()
    first_chunks = None
    cumulative = torch.zeros(n_samples, device=device)
    alive = torch.ones(n_samples, device=device)

    for k in range(horizon):
        chunks = policy.ddim_sample(state, n_steps=ddim_steps, use_ema=True)
        if k == 0:
            first_chunks = chunks

        delta_std_pred, r_std_pred, done_logit = wm(state, chunks)
        delta = delta_std_pred * stats["delta_std"] + stats["delta_mean"]
        r = r_std_pred * stats["reward_std"] + stats["reward_mean"]
        done_prob = torch.sigmoid(done_logit).clamp(0.0, 1.0)

        cumulative = cumulative + (gamma ** k) * r * alive
        alive = alive * (1.0 - done_prob)
        state = state + delta

    best = int(torch.argmax(cumulative).item())
    return first_chunks[best]


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy, obs_norm, action_norm, obs_keys, cfg = load_bc(args.bc_checkpoint, device)
    wm, stats = load_wm(args.wm_checkpoint, device)
    action_dim = len(action_norm.mins)
    pred_horizon = cfg["pred_horizon"]
    action_per_step = action_dim // pred_horizon

    env = make_env_from_hdf5(args.data_path or cfg.get("data_path"),
                             horizon=args.env_horizon)

    successes = 0
    t0 = time.time()
    for ep in range(args.n_episodes):
        obs_dict = env.reset()
        obs_flat = extract_obs(obs_dict, obs_keys)
        ep_success = False
        step_count = 0
        done = False
        t_ep = time.time()

        while not done and step_count < args.env_horizon:
            obs_norm_t = obs_norm.normalize(
                torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
            ).squeeze(0)

            with torch.no_grad():
                if args.n_samples <= 1:
                    chunk = policy.ddim_sample(obs_norm_t.unsqueeze(0), n_steps=args.ddim_steps,
                                               use_ema=True).squeeze(0)
                else:
                    chunk = plan_mpc(obs_norm_t, policy, wm, stats,
                                     args.n_samples, args.mpc_horizon, args.gamma,
                                     args.ddim_steps, device)

            action_flat = action_norm.unnormalize(chunk.unsqueeze(0)).squeeze(0).cpu().numpy()
            actions = action_flat.reshape(pred_horizon, action_per_step)
            for a in actions:
                a = np.clip(a, -1, 1)
                obs_dict, _, env_done, _ = env.step(a)
                step_count += 1
                if env._check_success():
                    ep_success = True
                    done = True
                    break
                if env_done or step_count >= args.env_horizon:
                    done = True
                    break
            obs_flat = extract_obs(obs_dict, obs_keys)

        successes += int(ep_success)
        print(f"ep {ep + 1:03d} | {time.time() - t_ep:.1f}s | steps {step_count} | success {ep_success}")

    env.close()
    rate = successes / args.n_episodes
    print(f"\nMPC success: {successes}/{args.n_episodes} = {rate:.1%} "
          f"| total {time.time() - t0:.1f}s | N={args.n_samples} H={args.mpc_horizon}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_checkpoint", type=str, required=True)
    parser.add_argument("--wm_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str,
                        default="/home/abhi/dsrl/datasets/transport/ph/low_dim_v141.hdf5",
                        help="env hdf5 (for env kwargs); overrides any path stored in ckpt")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=8,
                        help="MPC candidate count (1 = plain BC, no planning)")
    parser.add_argument("--mpc_horizon", type=int, default=3,
                        help="chunks of lookahead in the world model")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ddim_steps", type=int, default=10)
    parser.add_argument("--env_horizon", type=int, default=700)
    args = parser.parse_args()
    evaluate(args)

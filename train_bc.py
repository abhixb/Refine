import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from data.dataset import RoboMimicDataset
from model.diffusion import DiffusionPolicy


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RoboMimicDataset(args.data_path, pred_horizon=args.pred_horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    obs_dim = dataset.obs.shape[1]
    action_dim = dataset.actions.shape[1]

    policy = DiffusionPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
        n_timesteps=args.diffusion_steps,
        hidden_dims=tuple(args.hidden_dims),
        ema_decay=args.ema_decay,
    ).to(device)

    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda")

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        policy.model.train()
        epoch_loss = 0.0
        n_batches = 0

        for obs, action in loader:
            obs, action = obs.to(device), action.to(device)

            with autocast("cuda"):
                loss = policy.training_loss(action, obs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            policy.update_ema()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % args.log_every == 0:
            print(f"epoch {epoch:04d} | loss {avg_loss:.6f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "obs_normalizer": dataset.obs_normalizer.state_dict(),
                "action_normalizer": dataset.action_normalizer.state_dict(),
                "obs_keys": dataset.obs_keys,
                "config": vars(args),
            }
            path = os.path.join(args.save_dir, f"state_{epoch}.pt")
            torch.save(ckpt, path)
            print(f"saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints/bc")
    parser.add_argument("--task", type=str, default="lift")
    parser.add_argument("--pred_horizon", type=int, default=4)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[1024, 1024, 1024])
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()
    train(args)

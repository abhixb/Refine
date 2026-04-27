import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.normalize import MinMaxNormalizer

SINGLE_ARM_OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
BIMANUAL_OBS_KEYS = [
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
    "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos",
    "object",
]


def detect_obs_keys(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        available = list(f["data/demo_0/obs"].keys())
    if "robot1_eef_pos" in available:
        return BIMANUAL_OBS_KEYS
    return SINGLE_ARM_OBS_KEYS


def load_robomimic_hdf5(path, pred_horizon=4, obs_keys=None, hold_steps=0):
    if obs_keys is None:
        obs_keys = detect_obs_keys(path)

    obs_chunks = []
    action_chunks = []

    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.replace("demo_", "")))
        for demo_key in demos:
            demo = f["data"][demo_key]
            obs_parts = [demo["obs"][k][:] for k in obs_keys]
            obs = np.concatenate(obs_parts, axis=-1)
            actions = demo["actions"][:]

            if hold_steps > 0:
                obs = np.concatenate([obs, np.tile(obs[-1:], (hold_steps, 1))], axis=0)
                hold_action = actions[-1:].copy()
                n_arms = hold_action.shape[1] // 7
                for i in range(n_arms):
                    hold_action[:, i * 7 : i * 7 + 6] = 0.0
                actions = np.concatenate([actions, np.tile(hold_action, (hold_steps, 1))], axis=0)

            T = actions.shape[0]
            for t in range(T - pred_horizon + 1):
                obs_chunks.append(obs[t])
                action_chunks.append(actions[t:t + pred_horizon].reshape(-1))

    obs_all = np.stack(obs_chunks).astype(np.float32)
    action_all = np.stack(action_chunks).astype(np.float32)
    return obs_all, action_all, obs_keys


class RoboMimicDataset(Dataset):
    def __init__(self, hdf5_path, pred_horizon=4, hold_steps=0):
        obs_raw, action_raw, self.obs_keys = load_robomimic_hdf5(
            hdf5_path, pred_horizon, hold_steps=hold_steps
        )
        self.obs_normalizer = MinMaxNormalizer(obs_raw)
        self.action_normalizer = MinMaxNormalizer(action_raw)
        self.obs = self.obs_normalizer.normalize(obs_raw).astype(np.float32)
        self.actions = self.action_normalizer.normalize(action_raw).astype(np.float32)

        print(f"loaded {len(self)} samples | obs dim {self.obs.shape[1]} | action dim {self.actions.shape[1]} | keys {self.obs_keys}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return torch.tensor(self.obs[idx]), torch.tensor(self.actions[idx])

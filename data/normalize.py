import numpy as np
import torch


class MinMaxNormalizer:
    def __init__(self, data, eps=1e-6):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        self.mins = data.min(axis=0)
        self.maxs = data.max(axis=0)
        self.range = self.maxs - self.mins
        self.range[self.range < eps] = 1.0

    def normalize(self, x):
        if isinstance(x, np.ndarray):
            return 2.0 * (x - self.mins) / self.range - 1.0
        mins = torch.tensor(self.mins, dtype=x.dtype, device=x.device)
        range_ = torch.tensor(self.range, dtype=x.dtype, device=x.device)
        return 2.0 * (x - mins) / range_ - 1.0

    def unnormalize(self, x):
        if isinstance(x, np.ndarray):
            return (x + 1.0) / 2.0 * self.range + self.mins
        mins = torch.tensor(self.mins, dtype=x.dtype, device=x.device)
        range_ = torch.tensor(self.range, dtype=x.dtype, device=x.device)
        return (x + 1.0) / 2.0 * range_ + mins

    def state_dict(self):
        return {"mins": self.mins, "maxs": self.maxs, "range": self.range}

    def load_state_dict(self, d):
        self.mins = d["mins"]
        self.maxs = d["maxs"]
        self.range = d["range"]

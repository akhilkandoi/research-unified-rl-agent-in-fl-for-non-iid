"""Task CNNs (no batch norm, ReLU, 2x2 max pool) and deterministic initialization."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from fedrl.utils.flat import get_flat
from fedrl.utils.seeding import stable_seed

# Per-dataset constants (channels, image size, classes, normalization).
DATASET_SPECS: dict[str, dict] = {
    "mnist": {"in_channels": 1, "image_size": 28, "num_classes": 10, "mean": (0.1307,), "std": (0.3081,)},
    "fmnist": {"in_channels": 1, "image_size": 28, "num_classes": 10, "mean": (0.2860,), "std": (0.3530,)},
    "cifar10": {"in_channels": 3, "image_size": 32, "num_classes": 10,
                "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
}

_ARCH = {  # conv1 out, conv2 out, flatten size, hidden fc
    "mnist": (16, 32, 512, 64),
    "fmnist": (16, 32, 512, 64),
    "cifar10": (32, 64, 1600, 128),
}


class TaskCNN(nn.Module):
    """conv5 -> relu -> pool -> conv5 -> relu -> pool -> flatten -> fc -> relu -> fc."""

    def __init__(self, in_channels: int, c1: int, c2: int, flat: int, hidden: int, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, c1, 5)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.fc1 = nn.Linear(flat, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


def build_model(dataset: str) -> nn.Module:
    """Build the task CNN for 'mnist' | 'fmnist' | 'cifar10' (default PyTorch init, CPU)."""
    if dataset not in _ARCH:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {sorted(_ARCH)}.")
    c1, c2, flat, hidden = _ARCH[dataset]
    spec = DATASET_SPECS[dataset]
    return TaskCNN(spec["in_channels"], c1, c2, flat, hidden, spec["num_classes"])


def init_model_vec(dataset: str, seed: int) -> Tensor:
    """Flat initial parameter vector (CPU float32) identical for every method given (dataset, seed).

    Does not disturb the global torch RNG.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(stable_seed(seed, f"model_init::{dataset}"))
        model = build_model(dataset)
    return get_flat(model)
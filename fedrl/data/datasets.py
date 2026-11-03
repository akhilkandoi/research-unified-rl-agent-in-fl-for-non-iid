"""Dataset loading (MNIST, Fashion-MNIST, CIFAR-10) into normalized on-device tensors.

LEAK GUARD: ``get_official_test`` returns the official TEST set. Only
``fedrl/eval``, ``scripts/`` and ``tests/`` (plus ``fedrl/data/scenario.py``,
which builds evaluation-only personal test sets) may import it. A static test
(tests/test_no_test_leak.py) enforces this. Training, tuning and checkpoint
selection must use validation data only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from fedrl.utils.logging_utils import get_logger

log = get_logger(__name__)

from fedrl.models.cnn import DATASET_SPECS

DATASET_NAMES = ("mnist", "fmnist", "cifar10")
NUM_CLASSES = {k: int(DATASET_SPECS[k]["num_classes"]) for k in DATASET_NAMES}
NORM_STATS = {
    k: (tuple(float(v) for v in np.atleast_1d(DATASET_SPECS[k]["mean"])),
        tuple(float(v) for v in np.atleast_1d(DATASET_SPECS[k]["std"])))
    for k in DATASET_NAMES
}


@dataclass
class DatasetBundle:
    """Normalized float tensors for one dataset, all on ``device``."""

    name: str
    num_classes: int
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    device: torch.device


_CACHE: dict[tuple[str, str, str], DatasetBundle] = {}


def raw_to_tensors(data, targets, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw uint8 images (NHW or NHWC) and labels to normalized NCHW float tensors (CPU)."""
    if name not in NORM_STATS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASET_NAMES}.")
    mean, std = NORM_STATS[name]
    x = torch.as_tensor(np.asarray(data))
    if x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim == 4:
        x = x.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unexpected image tensor rank {x.ndim}")
    if x.shape[1] != len(mean):
        raise ValueError(f"{name}: expected {len(mean)} channels, got {x.shape[1]}")
    x = x.contiguous().float().div_(255.0)
    m = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
    x = ((x - m) / s).contiguous()
    y = torch.as_tensor(np.asarray(targets), dtype=torch.long)
    return x, y


def _load_torchvision(name: str, data_dir: str):
    from torchvision import datasets as tvd  # imported lazily: only needed for download

    cls = {"mnist": tvd.MNIST, "fmnist": tvd.FashionMNIST, "cifar10": tvd.CIFAR10}[name]
    tr = cls(root=data_dir, train=True, download=True)
    te = cls(root=data_dir, train=False, download=True)
    return tr, te


def load_dataset(name: str, data_dir: str, device) -> DatasetBundle:
    """Load ``name`` (mnist | fmnist | cifar10), cache under ``data_dir`` (local disk),
    normalize and move to ``device`` ONCE (also memoized per process)."""
    if name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASET_NAMES}.")
    dev = torch.device(device)
    key = (name, os.path.abspath(data_dir), str(dev))
    if key in _CACHE:
        return _CACHE[key]
    os.makedirs(data_dir, exist_ok=True)
    log.info("Loading dataset %s from %s", name, data_dir)
    tr, te = _load_torchvision(name, data_dir)
    tx, ty = raw_to_tensors(tr.data, tr.targets, name)
    ex, ey = raw_to_tensors(te.data, te.targets, name)
    bundle = DatasetBundle(
        name=name,
        num_classes=NUM_CLASSES[name],
        train_x=tx.to(dev),
        train_y=ty.to(dev),
        test_x=ex.to(dev),
        test_y=ey.to(dev),
        device=dev,
    )
    _CACHE[key] = bundle
    return bundle


def get_official_test(bundle: DatasetBundle) -> tuple[torch.Tensor, torch.Tensor]:
    """Official TEST set (x, y). Final evaluation only; see the module docstring."""
    return bundle.test_x, bundle.test_y
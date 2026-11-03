"""Shared test helpers: a small synthetic dataset (no download) and a CPU test config."""
from __future__ import annotations

import torch

from fedrl.config import load_config
from fedrl.data.datasets import DatasetBundle
from fedrl.data.pools import build_pools


def make_synthetic_bundle(n_train: int = 30000, n_test: int = 5000, num_classes: int = 10,
                          seed: int = 0, name: str = "mnist") -> DatasetBundle:
    g = torch.Generator().manual_seed(seed)
    ytr = (torch.arange(n_train) % num_classes)[torch.randperm(n_train, generator=g)]
    yte = (torch.arange(n_test) % num_classes)[torch.randperm(n_test, generator=g)]
    return DatasetBundle(
        name=name, num_classes=num_classes,
        train_x=torch.randn(n_train, 1, 2, 2, generator=g), train_y=ytr.long(),
        test_x=torch.randn(n_test, 1, 2, 2, generator=g), test_y=yte.long(),
        device=torch.device("cpu"),
    )


def make_cfg(extra: list[str] | None = None):
    return load_config(None, ["run.device=cpu", "data.global_val_size=500"] + list(extra or []))


def make_world(extra: list[str] | None = None):
    cfg = make_cfg(extra)
    bundle = make_synthetic_bundle()
    return cfg, bundle, build_pools(bundle, cfg)
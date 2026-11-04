"""Shared helpers for tests and scripts: synthetic data, patching, comparisons.

This module never imports ``get_official_test``; it only patches it by name.
"""
from __future__ import annotations

import contextlib
import importlib
import logging
import pkgutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import torch

import fedrl
from fedrl.data.datasets import DatasetBundle

logger = logging.getLogger(__name__)

_SHAPES = {"mnist": (1, 28, 28), "fmnist": (1, 28, 28), "cifar10": (3, 32, 32)}
_TIME_TOKENS = {"time", "sec", "secs", "seconds", "wall", "date", "timestamp", "elapsed"}


def make_synthetic_bundle(dataset: str = "mnist", device: Any = "cpu", n_train: int = 8000,
                          n_test: int = 3000, seed: int = 0) -> DatasetBundle:
    """Return a learnable class-template image dataset with the shapes of ``dataset``."""
    if dataset not in _SHAPES:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(_SHAPES)}")
    ch, h, w = _SHAPES[dataset]
    dev = torch.device(device)
    gen = torch.Generator().manual_seed(seed)
    templates = torch.randn(10, ch, h, w, generator=gen)

    def make(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = (torch.arange(n) % 10)[torch.randperm(n, generator=gen)]
        x = 0.8 * templates[y] + torch.randn(n, ch, h, w, generator=gen)
        return x.contiguous().to(dev), y.long().to(dev)

    tx, ty = make(n_train)
    sx, sy = make(n_test)
    return DatasetBundle(name=dataset, num_classes=10, train_x=tx, train_y=ty,
                         test_x=sx, test_y=sy, device=dev)


def _import_all_fedrl() -> None:
    for info in pkgutil.walk_packages(fedrl.__path__, "fedrl."):
        try:
            importlib.import_module(info.name)
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.warning("could not import %s: %s", info.name, exc)


@contextlib.contextmanager
def patch_everywhere(name: str, replacement: Callable[..., Any]) -> Iterator[None]:
    """Replace attribute ``name`` in every loaded ``fedrl.*`` module, restore on exit."""
    _import_all_fedrl()
    saved: list[tuple[Any, Any]] = []
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("fedrl") and mod is not None and hasattr(mod, name):
            saved.append((mod, getattr(mod, name)))
            setattr(mod, name, replacement)
    try:
        yield
    finally:
        for mod, original in saved:
            setattr(mod, name, original)


@contextlib.contextmanager
def install_synthetic_data() -> Iterator[Callable[..., DatasetBundle]]:
    """Make ``load_dataset`` return synthetic data (offline runs)."""
    cache: dict[tuple[str, str], DatasetBundle] = {}

    def fake_load_dataset(name: str, data_dir: Any = None, device: Any = "cpu") -> DatasetBundle:
        key = (name, str(device))
        if key not in cache:
            cache[key] = make_synthetic_bundle(name, device)
        return cache[key]

    with patch_everywhere("load_dataset", fake_load_dataset):
        yield fake_load_dataset


def _blocked(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("get_official_test was called: training/validation must never read the official test set")


def block_official_test() -> contextlib.AbstractContextManager:
    """Context in which any call to ``get_official_test`` raises AssertionError."""
    return patch_everywhere("get_official_test", _blocked)


def tensor_leaves(obj: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten nested dict/list/tuple containers into {path: float64 array}."""
    out: dict[str, np.ndarray] = {}
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj.detach().cpu().double().numpy()
    elif isinstance(obj, np.ndarray) and obj.dtype.kind in "biuf":
        out[prefix] = obj.astype(np.float64)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(tensor_leaves(v, f"{prefix}/{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(tensor_leaves(v, f"{prefix}/{i}"))
    return out


def compare_checkpoints(a: dict, b: dict, rtol: float = 1e-4, atol: float = 1e-5) -> tuple[int, list[str]]:
    """Compare policy tensors of two checkpoint dicts; returns (n_compared, problems)."""
    la, lb = tensor_leaves(a), tensor_leaves(b)
    keys = [k for k in la if "policy" in k.lower() and "optim" not in k.lower()]
    if not keys:
        keys = [k for k in la if "optim" not in k.lower() and "rng" not in k.lower()]
    problems: list[str] = []
    for k in keys:
        if k not in lb:
            problems.append(f"missing in second checkpoint: {k}")
        elif la[k].shape != lb[k].shape or not np.allclose(la[k], lb[k], rtol=rtol, atol=atol, equal_nan=True):
            problems.append(f"values differ: {k}")
    return len(keys), problems


def compare_run_logs(dir_a: Path, dir_b: Path, rtol: float = 1e-4, atol: float = 1e-5) -> list[str]:
    """Compare every CSV log of two run folders, ignoring wall-clock style columns."""
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    fa = sorted(p.relative_to(dir_a) for p in dir_a.rglob("*.csv"))
    fb = sorted(p.relative_to(dir_b) for p in dir_b.rglob("*.csv"))
    if fa != fb:
        return [f"csv file sets differ: {fa} vs {fb}"]
    if not fa:
        return ["no csv logs found in run folder"]
    problems: list[str] = []
    for rel in fa:
        da, db = pd.read_csv(dir_a / rel), pd.read_csv(dir_b / rel)
        if da.shape != db.shape:
            problems.append(f"{rel}: shape {da.shape} vs {db.shape}")
            continue
        for col in da.columns:
            if _TIME_TOKENS & set(str(col).lower().split("_")):
                continue
            if pd.api.types.is_numeric_dtype(da[col]) and pd.api.types.is_numeric_dtype(db[col]):
                if not np.allclose(da[col].to_numpy(float), db[col].to_numpy(float),
                                   rtol=rtol, atol=atol, equal_nan=True):
                    problems.append(f"{rel}: column {col} differs")
            elif not da[col].equals(db[col]):
                problems.append(f"{rel}: column {col} differs")
    return problems


def default_action(obs: dict, cfg: Any, alpha_value: float | None = None) -> dict:
    """First k eligible clients, default menu indices (largest feasible epochs if masked)."""
    elig, emask = obs["eligible"], obs["epochs_mask"]
    dev = obs["state"].device
    n, k = int(elig.numel()), int(cfg.fl.k)
    sel = torch.nonzero(elig).flatten()[:k].long()
    epochs = torch.full((n,), int(cfg.menus.default_epochs_idx), dtype=torch.long, device=dev)
    for i in sel.tolist():
        if not bool(emask[i, epochs[i]]):
            feasible = torch.nonzero(emask[i]).flatten()
            if feasible.numel() > 0:
                epochs[i] = feasible.max()

    def full(idx: int) -> torch.Tensor:
        return torch.full((n,), int(idx), dtype=torch.long, device=dev)

    act = {"selected": sel.to(dev), "lr_idx": full(cfg.menus.default_lr_idx), "epochs_idx": epochs,
           "batch_idx": full(cfg.menus.default_batch_idx), "alpha_idx": full(cfg.menus.default_alpha_idx)}
    if alpha_value is not None:
        act["alpha_value"] = torch.full((n,), float(alpha_value), device=dev)
    return act
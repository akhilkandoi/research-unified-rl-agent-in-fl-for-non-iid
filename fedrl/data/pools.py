"""Disjoint index pools built once from the official TRAIN set with a fixed pool seed (D1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fedrl.data.datasets import DatasetBundle
from fedrl.utils.logging_utils import get_logger
from fedrl.utils.seeding import make_rng

log = get_logger(__name__)


@dataclass
class Pools:
    """Index pools into ``bundle.train_x``: balanced global-val, client-train and client-val per class.

    Tensors live on the bundle device; numpy mirrors (used for sampling) are filled automatically.
    """

    global_val_idx: torch.Tensor
    train_by_class: list[torch.Tensor]
    val_by_class: list[torch.Tensor]
    global_val_np: np.ndarray | None = None
    train_np: list[np.ndarray] | None = None
    val_np: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.global_val_np is None:
            self.global_val_np = self.global_val_idx.detach().cpu().numpy().astype(np.int64)
        if self.train_np is None:
            self.train_np = [t.detach().cpu().numpy().astype(np.int64) for t in self.train_by_class]
        if self.val_np is None:
            self.val_np = [t.detach().cpu().numpy().astype(np.int64) for t in self.val_by_class]


def assert_pools_disjoint(pools: Pools) -> None:
    """Raise AssertionError unless global-val, client-train and client-val are pairwise disjoint."""
    gv = pools.global_val_np
    tr = np.concatenate(pools.train_np)
    va = np.concatenate(pools.val_np)
    for name, arr in (("global_val", gv), ("client_train", tr), ("client_val", va)):
        if len(np.unique(arr)) != len(arr):
            raise AssertionError(f"Pool '{name}' contains duplicate indices")
    for (na, a), (nb, b) in (
        (("global_val", gv), ("client_train", tr)),
        (("global_val", gv), ("client_val", va)),
        (("client_train", tr), ("client_val", va)),
    ):
        if np.intersect1d(a, b).size:
            raise AssertionError(f"Pools '{na}' and '{nb}' are not disjoint")


def build_pools(bundle: DatasetBundle, cfg) -> Pools:
    """Build the pools with ``data.pool_seed`` (identical for every scenario and method).

    Global-val: ``global_val_size // C`` per class (exactly balanced). The remainder of each class
    is split ``client_train_frac`` / rest into client-train and client-val.
    """
    C = bundle.num_classes
    labels = bundle.train_y.detach().cpu().numpy()
    rng = make_rng(int(cfg.data.pool_seed), "pools")
    gv_total = int(cfg.data.global_val_size)
    per_class = gv_total // C
    if per_class * C != gv_total:
        log.warning("global_val_size=%d not divisible by %d classes; using %d", gv_total, C, per_class * C)
    frac = float(cfg.data.client_train_frac)
    gv_list, tr_list, va_list = [], [], []
    for c in range(C):
        idx = np.flatnonzero(labels == c)
        if len(idx) <= per_class:
            raise ValueError(f"Class {c} has only {len(idx)} training samples; cannot take {per_class} for global-val")
        idx = rng.permutation(idx)
        gv_list.append(np.sort(idx[:per_class]))
        rest = idx[per_class:]
        n_tr = int(round(frac * len(rest)))
        tr_list.append(np.sort(rest[:n_tr]))
        va_list.append(np.sort(rest[n_tr:]))
    dev = bundle.device

    def to_t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.long, device=dev)

    pools = Pools(
        global_val_idx=to_t(np.concatenate(gv_list)),
        train_by_class=[to_t(a) for a in tr_list],
        val_by_class=[to_t(a) for a in va_list],
    )
    assert_pools_disjoint(pools)
    log.info(
        "Pools: global_val=%d, client_train=%d, client_val=%d",
        len(pools.global_val_np), sum(len(a) for a in tr_list), sum(len(a) for a in va_list),
    )
    return pools
"""Communication / computation costs, reference and normalization (MASTER 2.4)."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch


def n_transfers(cfg) -> int:
    """Transfers per chosen client: base (download+upload), +1 in after_aggregation mode."""
    extra = 1 if cfg.fl.blend_mode == "after_aggregation" else 0
    return int(cfg.clients.n_transfers_base) + extra


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().double().numpy()
    return np.asarray(x, dtype=np.float64)


def round_costs(chosen: Sequence[int], epochs_vals: Sequence[int], n, speed, bw,
                model_bytes: float, cfg) -> Tuple[float, float, float]:
    """Return (comm, comp, latency) summed/maxed over chosen clients.

    comm_i = model_bytes * n_transfers / bw_i ; comp_i = epochs_i * n_i / speed_i ;
    latency = max_i (comm_i + comp_i). Empty selection gives zeros.
    """
    ch = [int(i) for i in (chosen.tolist() if isinstance(chosen, torch.Tensor) else chosen)]
    ep = _np(list(epochs_vals))
    if len(ch) != len(ep):
        raise ValueError("chosen and epochs_vals must have the same length")
    if not ch:
        return 0.0, 0.0, 0.0
    idx = np.asarray(ch, dtype=np.int64)
    comm_i = float(model_bytes) * n_transfers(cfg) / _np(bw)[idx]
    comp_i = ep * _np(n)[idx] / _np(speed)[idx]
    return float(comm_i.sum()), float(comp_i.sum()), float((comm_i + comp_i).max())


def cost_reference(n, speed, bw, model_bytes: float, cfg) -> Tuple[float, float]:
    """(comm_ref, comp_ref) = k * mean over ALL clients of the cost at median settings (epochs 2)."""
    k = int(cfg.fl.k)
    e_med = cfg.menus.epochs[int(cfg.menus.default_epochs_idx)]
    comm_ref = k * float(np.mean(float(model_bytes) * n_transfers(cfg) / _np(bw)))
    comp_ref = k * float(np.mean(e_med * _np(n) / _np(speed)))
    return comm_ref, comp_ref


def normalize_costs(comm: float, comp: float, refs: Tuple[float, float], cfg) -> Tuple[float, float]:
    """Divide by the references and clip to [0, reward.cost_clip]."""
    comm_ref, comp_ref = refs
    if comm_ref <= 0 or comp_ref <= 0:
        raise ValueError(f"Cost references must be positive, got {refs}")
    clip = float(cfg.reward.cost_clip)
    return (float(np.clip(comm / comm_ref, 0.0, clip)),
            float(np.clip(comp / comp_ref, 0.0, clip)))
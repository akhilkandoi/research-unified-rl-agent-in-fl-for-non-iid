"""Battery model, eligibility mask and epochs mask (MASTER 2.4)."""
from __future__ import annotations

from typing import Sequence, Union

import torch

Number = Union[float, torch.Tensor]

# Tolerance for float32 boundary cases in "battery - drain >= min_battery".
ELIGIBILITY_TOL = 1e-6


def drain_coef(cfg) -> float:
    """Drain per (epoch * sample): drain_ref / (drain_ref_epochs * drain_ref_n)."""
    c = cfg.clients
    return float(c.drain_ref) / (float(c.drain_ref_epochs) * float(c.drain_ref_n))


def drain(epochs: Number, n: Number, cfg) -> Number:
    """Battery drained by training `epochs` epochs on `n` samples (scalar or tensor)."""
    return drain_coef(cfg) * epochs * n


def eligible_mask(battery: torch.Tensor, n: torch.Tensor, cfg) -> torch.Tensor:
    """BoolTensor[N]: client can afford the smallest epochs option and stay >= min_battery."""
    e_min = min(cfg.menus.epochs)
    return (battery - drain(e_min, n, cfg)) >= (float(cfg.clients.min_battery) - ELIGIBILITY_TOL)


def epochs_mask(battery: torch.Tensor, n: torch.Tensor, cfg) -> torch.Tensor:
    """BoolTensor[N, E]: True where the epochs option is affordable. lr/batch/alpha never masked."""
    min_b = float(cfg.clients.min_battery) - ELIGIBILITY_TOL
    cols = [(battery - drain(e, n, cfg)) >= min_b for e in cfg.menus.epochs]
    return torch.stack(cols, dim=1)


class Battery:
    """Per-client battery levels with drain for chosen clients and recharge for all."""

    def __init__(self, level0: torch.Tensor, n: torch.Tensor, cfg) -> None:
        self.cfg = cfg
        self.level = level0.detach().clone().to(torch.float32)
        self.n = n

    def eligible(self) -> torch.Tensor:
        return eligible_mask(self.level, self.n, self.cfg)

    def epochs_mask(self) -> torch.Tensor:
        return epochs_mask(self.level, self.n, self.cfg)

    def apply_drain(self, chosen: Sequence[int], epochs_vals: Sequence[int]) -> None:
        """Drain chosen clients by drain(epochs, n_i). Level never goes below 0."""
        if len(chosen) != len(epochs_vals):
            raise ValueError("chosen and epochs_vals must have the same length")
        if len(chosen) == 0:
            return
        dev = self.level.device
        idx = torch.as_tensor(list(chosen), dtype=torch.long, device=dev)
        ep = torch.as_tensor(list(epochs_vals), dtype=self.level.dtype, device=dev)
        self.level[idx] -= drain(ep, self.n[idx].to(self.level.dtype), self.cfg)
        self.level.clamp_(min=0.0)

    def recharge(self) -> None:
        """Every client recharges by clients.recharge, capped at 1.0."""
        self.level.add_(float(self.cfg.clients.recharge)).clamp_(max=1.0)
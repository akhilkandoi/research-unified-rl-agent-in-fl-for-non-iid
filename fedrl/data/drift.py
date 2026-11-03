"""Drift logic (MASTER 2.3 / D3). Pure functions: no hidden state, fully deterministic from the seed.

Rounds are 1-based. Abrupt: at round ``drift.abrupt_round`` every client redraws p ~ Dirichlet(a).
Gradual: p(t) = (1-lam) p_old + lam p_new, lam = clip((t - start + 1) / R, 0, 1), for
t in [start, start + R - 1]; train/val indices are resampled in every transition round.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedrl.data.partition import ClientDraw, draw_client_data, sample_proportions
from fedrl.utils.seeding import make_rng

DRIFT_KINDS = ("none", "abrupt", "gradual")


def check_drift_kind(drift: str) -> None:
    if drift not in DRIFT_KINDS:
        raise ValueError(f"Unknown drift '{drift}'. Choose from {DRIFT_KINDS}.")


def drift_due(drift: str, round_t: int, cfg) -> bool:
    """True if data changes at the START of round ``round_t``."""
    check_drift_kind(drift)
    d = cfg.drift
    if drift == "none":
        return False
    if drift == "abrupt":
        return int(round_t) == int(d.abrupt_round)
    start = int(d.gradual_start)
    return start <= int(round_t) <= start + int(d.gradual_rounds) - 1


def gradual_lambda(round_t: int, cfg) -> float:
    """lam = clip((t - start + 1) / R, 0, 1)."""
    d = cfg.drift
    lam = (int(round_t) - int(d.gradual_start) + 1) / float(d.gradual_rounds)
    return float(min(1.0, max(0.0, lam)))


def target_proportions(seed: int, a: float, n_clients: int, num_classes: int) -> np.ndarray:
    """The post-drift proportions p_new [N, C], drawn once per scenario from stream 'drift_target'."""
    return sample_proportions(make_rng(seed, "drift_target"), a, num_classes, n_clients)


def mixing_lambda(drift: str, round_t: int, cfg) -> float:
    """Interpolation weight toward p_new in effect during round ``round_t``."""
    check_drift_kind(drift)
    if drift == "none":
        return 0.0
    if drift == "abrupt":
        return 1.0 if int(round_t) >= int(cfg.drift.abrupt_round) else 0.0
    return gradual_lambda(round_t, cfg)


def proportions_at(drift: str, round_t: int, p_old: np.ndarray, p_new: np.ndarray, cfg) -> np.ndarray:
    """Proportions [N, C] in effect during round ``round_t`` (no sampling involved)."""
    lam = mixing_lambda(drift, round_t, cfg)
    return (1.0 - lam) * np.asarray(p_old) + lam * np.asarray(p_new)


@dataclass
class DriftUpdate:
    """Result of one drift step. ``clients`` is None when nothing changed."""

    changed: bool
    proportions: np.ndarray
    clients: list[ClientDraw] | None


def apply_drift(drift: str, round_t: int, seed: int, p_old: np.ndarray, p_new: np.ndarray,
                sizes: np.ndarray, pools, cfg) -> DriftUpdate:
    """Pure drift step: returns the new proportions and, if drift is due this round, freshly
    resampled train / val / personal-val indices drawn from the NEW proportions."""
    props = proportions_at(drift, round_t, p_old, p_new, cfg)
    if not drift_due(drift, round_t, cfg):
        return DriftUpdate(False, props, None)
    tag = f"drift_r{int(round_t)}"
    draws = [draw_client_data(seed, i, int(sizes[i]), props[i], pools, cfg, tag) for i in range(len(sizes))]
    return DriftUpdate(True, props, draws)
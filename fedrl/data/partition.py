"""Client sizes, Dirichlet label proportions, index drawing and personal-validation sets (MASTER 2.2)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fedrl.utils.logging_utils import get_logger
from fedrl.utils.seeding import make_rng

log = get_logger(__name__)


def sample_client_sizes(rng: np.random.Generator, n_clients: int, cfg) -> np.ndarray:
    """Clipped lognormal sizes: mu = ln(mean) - sigma^2/2, clipped to [size_min, size_max], int."""
    c = cfg.clients
    sigma = float(c.size_sigma)
    mu = math.log(float(c.size_mean)) - sigma ** 2 / 2.0
    raw = rng.lognormal(mean=mu, sigma=sigma, size=int(n_clients))
    return np.clip(np.rint(raw), int(c.size_min), int(c.size_max)).astype(np.int64)


def _dirichlet_one(rng: np.random.Generator, a: float, num_classes: int) -> np.ndarray:
    """One Dirichlet(a*1) draw, guarded against NaN / all-zero results at tiny ``a``."""
    for _ in range(10):
        p = rng.dirichlet(np.full(num_classes, float(a)))
        if np.all(np.isfinite(p)) and p.sum() > 0:
            return p / p.sum()
    p = np.zeros(num_classes)
    p[int(rng.integers(num_classes))] = 1.0
    return p


def sample_proportions(rng: np.random.Generator, a: float, num_classes: int,
                       n_clients: int | None = None) -> np.ndarray:
    """Dirichlet(a*ones(C)) class proportions: shape [C], or [n_clients, C] if ``n_clients`` is given."""
    if a <= 0:
        raise ValueError(f"Dirichlet concentration a must be > 0, got {a}")
    if n_clients is None:
        return _dirichlet_one(rng, a, num_classes)
    return np.stack([_dirichlet_one(rng, a, num_classes) for _ in range(int(n_clients))])


def draw_counts(rng: np.random.Generator, n: int, p: np.ndarray) -> np.ndarray:
    """Multinomial(n, p) class counts."""
    p = np.asarray(p, dtype=np.float64)
    return rng.multinomial(int(n), p / p.sum()).astype(np.int64)


def cap_and_redistribute(rng: np.random.Generator, counts: np.ndarray, p: np.ndarray,
                         caps: np.ndarray, what: str = "pool") -> np.ndarray:
    """Cap ``counts`` at ``caps``; redistribute the excess over classes with room, weighted by ``p``
    (uniform over classes with room if all of them have p = 0). Logs a warning."""
    counts = counts.astype(np.int64).copy()
    p = np.asarray(p, dtype=np.float64)
    log.warning("%s: class pool too small for requested counts; capping and redistributing", what)
    while True:
        over = counts > caps
        if not over.any():
            break
        excess = int((counts - caps)[over].sum())
        counts[over] = caps[over]
        room = caps - counts
        avail = room > 0
        if not avail.any():
            log.warning("%s: total pool capacity exhausted; %d samples dropped", what, excess)
            break
        w = p * avail
        if w.sum() <= 0:
            w = avail.astype(np.float64)
        counts += rng.multinomial(excess, w / w.sum()).astype(np.int64)
    return counts


def draw_indices(rng: np.random.Generator, n: int, p: np.ndarray,
                 pool_by_class: list[np.ndarray], what: str = "client") -> np.ndarray:
    """Draw ``n`` indices following proportions ``p``; no repetition inside the call
    (pools hold unique indices). Shuffled int64 array."""
    caps = np.array([len(x) for x in pool_by_class], dtype=np.int64)
    counts = draw_counts(rng, n, p)
    if (counts > caps).any():
        counts = cap_and_redistribute(rng, counts, p, caps, what)
    parts = [rng.choice(pool_by_class[c], size=int(counts[c]), replace=False)
             for c in range(len(pool_by_class)) if counts[c] > 0]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return rng.permutation(np.concatenate(parts)).astype(np.int64)


def build_personal_val(rng: np.random.Generator, val_idx: np.ndarray,
                       global_val_idx: np.ndarray, own_frac: float) -> np.ndarray:
    """Personal validation set of the same size as ``val_idx``: ``own_frac`` from the client's own
    distribution (first part of its shuffled client-val draw) + the rest from the balanced global-val set."""
    m = len(val_idx)
    n_own = int(round(float(own_frac) * m))
    n_glob = min(m - n_own, len(global_val_idx))
    glob = rng.choice(global_val_idx, size=n_glob, replace=False)
    return rng.permutation(np.concatenate([val_idx[:n_own], glob])).astype(np.int64)


def val_size(n: int, cfg) -> int:
    """Client validation size: round(val_frac * n), at least val_min."""
    return max(int(cfg.clients.val_min), int(round(float(cfg.clients.val_frac) * int(n))))


@dataclass
class ClientDraw:
    """Index draws for one client (numpy int64 arrays)."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    personal_val_idx: np.ndarray


def draw_client_data(seed: int, client_id: int, n: int, props: np.ndarray, pools, cfg,
                     tag: str = "init") -> ClientDraw:
    """Deterministically draw train / val / personal-val indices for one client.

    ``tag`` ("init" or "drift_r<t>") is part of every RNG stream name, so initial data and every
    drift redraw are reproducible and independent.
    """
    train = draw_indices(make_rng(seed, f"{tag}_train_{client_id}"), n, props, pools.train_np, "client-train")
    val = draw_indices(make_rng(seed, f"{tag}_val_{client_id}"), val_size(n, cfg), props,
                       pools.val_np, "client-val")
    pval = build_personal_val(make_rng(seed, f"{tag}_pval_{client_id}"), val, pools.global_val_np,
                              float(cfg.data.personal_val_own_frac))
    return ClientDraw(train, val, pval)
"""Pilot go/no-go logic: plateau rule (D11) and pass criteria (D12). Pure functions, no I/O."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from fedrl.eval.metrics import selection_summary

REMEDIES = [
    "1. per-client credit assignment",
    "2. smaller action space",
    "3. reward rescaling or normalization",
    "4. longer training",
    "5. different memory",
]


def moving_average(x: np.ndarray | Sequence[float], window: int = 5) -> np.ndarray:
    """Trailing moving average along the last axis (expanding window at the start)."""
    a = np.asarray(x, dtype=float)
    out = np.empty_like(a)
    for j in range(a.shape[-1]):
        lo = max(0, j - window + 1)
        out[..., j] = a[..., lo:j + 1].mean(axis=-1)
    return out


def std_error(x: Sequence[float]) -> float:
    """Standard error of the mean (ddof=1); 0 for fewer than two values."""
    v = np.asarray(x, dtype=float)
    return float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0


def plateau_index(curves: np.ndarray, window: int = 5) -> tuple[int, float]:
    """D11: earliest point index p such that the 5-point moving average of the seed-mean curve stays
    within one standard error (across seeds) of its final value for all q >= p. curves: [S, P]."""
    c = np.atleast_2d(np.asarray(curves, dtype=float))
    ma = moving_average(c, window)
    mean = ma.mean(axis=0)
    se = std_error(ma[:, -1])
    ok = np.abs(mean - mean[-1]) <= se
    p = len(mean) - 1
    while p > 0 and ok[p - 1]:
        p -= 1
    return p, se


def sweep_plateau(curves_by_u: dict[int, np.ndarray], min_tail: float = 0.2, window: int = 5) -> dict:
    """Plateau analysis per update frequency; 'selected' = smallest value that plateaus (or None)."""
    info: dict[int, dict] = {}
    for u, curves in curves_by_u.items():
        p, se = plateau_index(curves, window)
        n_points = np.atleast_2d(curves).shape[1]
        need_tail = max(1, math.ceil(min_tail * n_points))
        info[int(u)] = {"index": int(p), "points": int(n_points), "se": float(se),
                        "plateaus": bool(n_points - p >= need_tail)}
    plateaus = sorted(u for u, v in info.items() if v["plateaus"])
    return {"per_u": info, "selected": plateaus[0] if plateaus else None}


def criterion_a(curves: np.ndarray, edge_frac: float = 0.2) -> dict:
    """(a) mean validation return of the last 20% exceeds the first 20% by more than its SE across seeds."""
    c = np.atleast_2d(np.asarray(curves, dtype=float))
    n_points = c.shape[1]
    if n_points < 2:
        return {"pass": False, "value": float("nan"), "detail": "fewer than 2 validation points"}
    n_edge = max(1, math.ceil(edge_frac * n_points))
    diff = c[:, -n_edge:].mean(axis=1) - c[:, :n_edge].mean(axis=1)
    mean, se = float(diff.mean()), std_error(diff)
    return {"pass": bool(mean > se and mean > 0), "value": mean,
            "detail": f"last-minus-first = {mean:.4f}, SE across seeds = {se:.4f}"}


def criterion_b(alphas: Sequence[Sequence[float]], low: float = 0.2, high: float = 0.8,
                min_std: float = 0.01, window: int = 5) -> dict:
    """(b) per seed: moving-average mean alpha of chosen clients inside [low, high] and std over episodes > min_std."""
    rows, ok_all = [], True
    for s, a in enumerate(alphas):
        v = np.asarray(a, dtype=float)
        if v.size == 0:
            rows.append(f"seed{s}: no alpha data")
            ok_all = False
            continue
        ma = moving_average(v, window)
        inside = bool(ma.min() >= low and ma.max() <= high)
        sd = float(v.std())
        ok = inside and sd > min_std
        ok_all &= ok
        rows.append(f"seed{s}: range=[{ma.min():.3f},{ma.max():.3f}] std={sd:.4f} -> {'ok' if ok else 'bad'}")
    return {"pass": bool(ok_all and len(alphas) > 0), "value": None, "detail": "; ".join(rows)}


def criterion_c(episode_counts: Sequence[Sequence[int]], min_coverage: float = 0.8, max_ratio: float = 3.0) -> dict:
    """(c) mean per-episode coverage >= 80% and no client above 3x its uniform share (pooled picks)."""
    if len(episode_counts) == 0:
        return {"pass": False, "value": None, "detail": "no episodes"}
    cov = float(np.mean([selection_summary(c)["coverage"] for c in episode_counts]))
    pooled = np.sum([np.asarray(c, dtype=float) for c in episode_counts], axis=0)
    ratio = selection_summary(pooled)["max_ratio"]
    ok = cov >= min_coverage and ratio <= max_ratio
    return {"pass": bool(ok), "value": {"coverage": cov, "max_ratio": ratio},
            "detail": f"mean coverage = {cov:.3f} (need >= {min_coverage}), max pick ratio = {ratio:.2f} (need <= {max_ratio})"}


def evaluate_pilot_criteria(curves: np.ndarray, alphas: Sequence[Sequence[float]],
                            episode_counts: Sequence[Sequence[int]]) -> dict:
    """All three D12 criteria plus the overall verdict."""
    res = {"a": criterion_a(curves), "b": criterion_b(alphas), "c": criterion_c(episode_counts)}
    res["all_pass"] = all(res[k]["pass"] for k in ("a", "b", "c"))
    return res


def format_criteria(res: dict) -> str:
    """Printable PASS/FAIL report; on failure lists the ordered remedies. Never adds episodes."""
    names = {"a": "(a) learning progress", "b": "(b) alpha in [0.2,0.8] and not constant", "c": "(c) selection coverage/balance"}
    lines = [f"  {'PASS' if res[k]['pass'] else 'FAIL'}  {names[k]}: {res[k]['detail']}" for k in ("a", "b", "c")]
    lines.append(f"  OVERALL: {'PASS' if res['all_pass'] else 'FAIL'}")
    if not res["all_pass"]:
        lines.append("  Ordered remedies (episodes are never added automatically):")
        lines += [f"    {r}" for r in REMEDIES]
    return "\n".join(lines)
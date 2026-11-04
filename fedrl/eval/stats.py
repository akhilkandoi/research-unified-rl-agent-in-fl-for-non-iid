"""Statistics: paired bootstrap, per-seed summaries and method comparisons (MASTER 2.13)."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from fedrl.methods import all_method_names, is_rl
from fedrl.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_METRICS = ("global_test_acc", "pers_test_acc", "worst10_acc", "min_acc", "comm", "comp",
                   "latency", "rounds_to_target", "auc", "drift_recovery")
LOWER_IS_BETTER = {"comm", "comp", "latency", "rounds_to_target", "drift_recovery"}


def non_rl_methods() -> list[str]:
    """All non-RL baseline names from the methods registry."""
    return [m for m in all_method_names() if not is_rl(m)]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int, ci: float,
                     rng: np.random.Generator) -> dict:
    """Paired bootstrap of mean(a - b). Pairs with NaN are dropped; arrays must be matched."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"paired_bootstrap needs matched 1-D arrays, got {a.shape} vs {b.shape}")
    d = (a - b)[np.isfinite(a) & np.isfinite(b)]
    n = d.size
    if n == 0:
        raise ValueError("paired_bootstrap: no valid pairs")
    means = d[rng.integers(0, n, size=(int(n_resamples), n))].mean(axis=1)
    tail = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return {"mean_diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_pairs": int(n)}


def summarize(df: pd.DataFrame, metrics: Sequence[str] = DEFAULT_METRICS,
              by: Sequence[str] = ("dataset", "method", "variant", "a", "drift")) -> pd.DataFrame:
    """Mean/std across TRAINING seeds (each seed first averaged over evaluation seeds) + per-seed values."""
    metrics = [m for m in metrics if m in df.columns]
    d = df.copy()
    for m in metrics:
        d[m] = pd.to_numeric(d[m], errors="coerce")
    keys = list(by)
    seed_level = d.groupby(keys + ["train_seed"], dropna=False)[metrics].mean().reset_index()
    rows = []
    for key, g in seed_level.groupby(keys, dropna=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["n_seeds"] = int(g.shape[0])
        for m in metrics:
            v = g[m]
            row[f"{m}_mean"] = float(v.mean()) if v.notna().any() else float("nan")
            row[f"{m}_std"] = float(v.std(ddof=1)) if v.notna().sum() > 1 else 0.0
            row[f"{m}_per_seed"] = {int(s): (None if pd.isna(x) else float(x))
                                    for s, x in zip(g["train_seed"], v)}
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_baseline(val_scores: dict[str, float], candidates: Iterable[str] | None = None) -> str:
    """Best non-RL baseline by VALIDATION score (higher is better); ties broken alphabetically."""
    cands = list(candidates) if candidates is not None else non_rl_methods()
    avail = {m: s for m, s in val_scores.items() if m in cands and np.isfinite(s)}
    if not avail:
        raise ValueError("no validation scores available for any baseline candidate")
    return sorted(avail, key=lambda m: (-avail[m], m))[0]


def _verdict(row: dict, higher_is_better: bool) -> str:
    if not row["excludes_zero"]:
        return "tie"
    good = row["mean_diff"] > 0 if higher_is_better else row["mean_diff"] < 0
    return "better" if good else "worse"


def compare_methods(df: pd.DataFrame, metric: str = "pers_test_acc", ref: str = "joint",
                    variant: str = "default", dataset: str | None = None,
                    best_baseline: str | None = None,
                    stacked: Sequence[str] = ("stacked_shared", "stacked_specialized"),
                    baselines: Sequence[str] | None = None, higher_is_better: bool | None = None,
                    n_resamples: int = 10000, ci: float = 0.95, seed: int = 0,
                    pooled: bool = True, print_count: bool = True) -> pd.DataFrame:
    """ref minus {stacked agents, best baseline, every baseline} per scenario (+ pooled 'ALL').

    Pairing unit = (a, drift, evaluation seed); training seeds are averaged first, so the pairs are
    matched on scenario seeds. best_baseline must come from VALIDATION (select_best_baseline).
    'better' only if the bootstrap CI excludes zero. Rounds inside runs are never resampled.
    """
    if higher_is_better is None:
        higher_is_better = metric not in LOWER_IS_BETTER
    d = df[df["variant"] == variant].copy()
    if dataset is not None:
        d = d[d["dataset"] == dataset]
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    cell = d.groupby(["method", "a", "drift", "eval_seed"])[metric].mean()
    present = set(d["method"].unique())
    if ref not in present:
        raise ValueError(f"reference method '{ref}' has no results for variant '{variant}'")
    base_list = [b for b in (baselines if baselines is not None else non_rl_methods()) if b in present]
    plan: list[tuple[str, str]] = [(s, "stacked") for s in stacked if s in present]
    if best_baseline is not None and best_baseline in present:
        plan.append((best_baseline, "best_baseline"))
    plan += [(b, "baseline") for b in base_list]
    rng = np.random.default_rng(seed)
    scen = sorted({(a, dr) for _, a, dr, _ in cell.index})
    groups = [((a, dr), [(a, dr)]) for a, dr in scen]
    if pooled and len(scen) > 1:
        groups.append((("ALL", "ALL"), scen))
    rows = []
    for other, role in plan:
        for (a, dr), members in groups:
            ra = cell.loc[ref]
            rb = cell.loc[other] if other in cell.index.get_level_values(0) else None
            if rb is None:
                continue
            sel = lambda s: s[[(x, y) in members for x, y, _ in s.index]]  # noqa: E731
            pair = pd.concat([sel(ra).rename("ref"), sel(rb).rename("other")], axis=1, join="inner").dropna()
            if pair.empty:
                logger.warning("no matched pairs for %s vs %s at %s/%s", ref, other, a, dr)
                continue
            res = paired_bootstrap(pair["ref"].to_numpy(), pair["other"].to_numpy(), n_resamples, ci, rng)
            row = {"metric": metric, "comparison": f"{ref} - {other}", "role": role, "other": other,
                   "a": a, "drift": dr, **res}
            row["verdict"] = _verdict(row, higher_is_better)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["n_comparisons"] = len(out)
    logger.info("compare_methods(%s): %d comparisons", metric, len(out))
    if print_count:
        logger.info("[stats] metric=%s: %d comparisons made (paired bootstrap, %d resamples, %d%% CI)",
                    metric, len(out), n_resamples, int(ci * 100))
    return out
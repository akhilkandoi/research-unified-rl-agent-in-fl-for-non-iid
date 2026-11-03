"""Pure metric helpers and result-file naming for evaluation (no environment access)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

RESULTS_SUBDIR = "results"
VALIDATION_FILE = "validation.json"


def to_jsonable(obj: Any) -> Any:
    """Recursively convert tensors / numpy objects to JSON-safe Python (NaN/inf -> None)."""
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def result_path(run_dir: str | Path, spec: Any, eval_seed: int, tag: str = "") -> Path:
    """Path of the single result JSON for (run folder, scenario spec, evaluation seed, tag)."""
    name = f"a{spec.a:g}_{spec.drift}_s{spec.seed}_e{eval_seed}"
    if tag:
        name += f"_{tag}"
    return Path(run_dir) / RESULTS_SUBDIR / f"{name}.json"


def validation_path(run_dir: str | Path) -> Path:
    """Path of the validation summary of a (baseline) run folder."""
    return Path(run_dir) / VALIDATION_FILE


def rounds_to_target(curve: Sequence[float], target: float | None) -> int | None:
    """First 1-based round whose accuracy is >= target; None if never (D15)."""
    if target is None:
        return None
    for r, acc in enumerate(curve, start=1):
        if acc >= target:
            return r
    return None


def area_under_curve(curve: Sequence[float]) -> float:
    """Normalised area under the per-round accuracy curve (= mean accuracy over rounds)."""
    if len(curve) == 0:
        raise ValueError("area_under_curve: empty curve")
    return float(np.mean(np.asarray(curve, dtype=float)))


def drift_window(cfg: Any, drift: str) -> tuple[int, int] | None:
    """(first round, last round) of the drift event, 1-based; None for 'none'."""
    if drift == "none":
        return None
    if drift == "abrupt":
        r = int(cfg.drift.abrupt_round)
        return (r, r)
    if drift == "gradual":
        s = int(cfg.drift.gradual_start)
        return (s, s + int(cfg.drift.gradual_rounds) - 1)
    raise ValueError(f"unknown drift '{drift}'")


def drift_recovery_time(curve: Sequence[float], start_round: int, end_round: int | None = None) -> int | None:
    """Rounds after the drift start needed to regain the pre-drift accuracy.

    Pre-drift accuracy = curve value of round ``start_round - 1``. Returns 0 if accuracy never
    falls below it from ``start_round`` on, None if it dips and never recovers (or no pre-drift
    round exists). Recovery must happen at or after ``end_round`` (gradual drift must be over).
    """
    c = np.asarray(curve, dtype=float)
    end = start_round if end_round is None else end_round
    if start_round < 2 or start_round > len(c):
        return None
    pre = c[start_round - 2]
    after = c[start_round - 1:]
    below = np.where(after < pre)[0]
    if below.size == 0:
        return 0
    first_dip_round = start_round + int(below[0])
    for r in range(max(end, first_dip_round), len(c) + 1):
        if c[r - 1] >= pre:
            return r - start_round
    return None


def client_accuracy_stats(accs: Sequence[float], percentile: float = 10.0) -> dict[str, float]:
    """Mean, std, worst-decile (np.percentile) and minimum of per-client accuracies."""
    a = np.asarray(accs, dtype=float)
    if a.size == 0:
        raise ValueError("client_accuracy_stats: no clients")
    return {"mean": float(a.mean()), "std": float(a.std()),
            "worst": float(np.percentile(a, percentile)), "min": float(a.min())}


def selection_summary(counts: Sequence[int]) -> dict[str, float]:
    """Coverage (fraction of clients picked at least once) and max share / uniform share."""
    c = np.asarray(counts, dtype=float)
    total = c.sum()
    if c.size == 0 or total <= 0:
        return {"coverage": 0.0, "max_ratio": 0.0}
    return {"coverage": float((c > 0).mean()), "max_ratio": float(c.max() / (total / c.size))}
"""Tolerant readers for Part 6 logs (val_log.csv, train_log.csv)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EPISODE_COLS = ("episode", "ep")
VAL_COLS = ("val_return", "validation_return", "mean_val_return", "return", "mean_return")
ALPHA_COLS = ("mean_alpha", "alpha_mean", "alpha")


def _pick(df: pd.DataFrame, cands: tuple[str, ...], what: str, path: Path) -> str:
    for c in cands:
        if c in df.columns:
            return c
    raise ValueError(f"{path}: no {what} column among {cands}; found {list(df.columns)}. "
                     f"Edit the *_COLS tuples in fedrl/eval/logs.py to match Part 6.")


def read_val_curve(run_dir: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(episodes, validation returns) from val_log.csv (mean if several rows per episode); None if absent."""
    p = Path(run_dir) / "val_log.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    e, v = _pick(df, EPISODE_COLS, "episode", p), _pick(df, VAL_COLS, "validation return", p)
    g = df.groupby(e)[v].mean()
    return g.index.to_numpy(), g.to_numpy(dtype=float)


def read_alpha_curve(run_dir: str | Path) -> np.ndarray | None:
    """Mean alpha of chosen clients per logged training row; None if absent."""
    p = Path(run_dir) / "train_log.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    return df[_pick(df, ALPHA_COLS, "mean alpha", p)].dropna().to_numpy(dtype=float)
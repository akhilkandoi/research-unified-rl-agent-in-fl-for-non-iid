"""Build tables, plots and statements from the result JSONs of all finished evaluation jobs."""
# ===== EDIT HERE =====
RESULTS_ROOT = None               # None = runs dir from config / FEDRL_RUNS_DIR / ./runs
OUT_DIR = "analysis_out"          # output folder for tables, plots, statements
CONFIG = "configs/default.yaml"
OVERRIDES = []                    # extra "key=value" overrides (e.g. ["run.runs_dir=./runs"])
REF = "joint"                     # reference method of all comparisons
PRIMARY_METRIC = "pers_test_acc"  # metric used for the win/tie/lose statements
COMPARE_METRICS = ["pers_test_acc", "global_test_acc", "worst10_acc"]  # metrics compared with bootstrap
FOCUS_A = 0.2                     # skew value used for E3/drift/heatmap plots
INCLUDE_PARAM_TABLE = True        # parameter-count table (builds the policies once)
SEED = 0                          # bootstrap RNG seed
# ===== END EDIT =====

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedrl.config import load_config
from fedrl.eval.analysis import resolve_runs_dir, run_analysis


def main(a) -> int:
    cfg = load_config(a.config, a.overrides)
    root = Path(a.root) if a.root else resolve_runs_dir(cfg)
    res = run_analysis(root, a.out_dir, cfg, a.ref, a.primary_metric, a.compare_metrics,
                       int(cfg.eval.bootstrap_resamples), float(cfg.eval.ci), a.param_table, a.focus_a, a.seed)
    print(f"Wrote {len(res['files'])} table files and plots to {res['out_dir']}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyse results (all flags optional; defaults = EDIT HERE).")
    ap.add_argument("--root", default=RESULTS_ROOT)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--set", action="append", default=list(OVERRIDES), dest="overrides")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--primary-metric", default=PRIMARY_METRIC)
    ap.add_argument("--compare-metrics", nargs="*", default=list(COMPARE_METRICS))
    ap.add_argument("--focus-a", type=float, default=FOCUS_A)
    ap.add_argument("--no-param-table", dest="param_table", action="store_false", default=INCLUDE_PARAM_TABLE)
    ap.add_argument("--seed", type=int, default=SEED)
    sys.exit(main(ap.parse_args()))
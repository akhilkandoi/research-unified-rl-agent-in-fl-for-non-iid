"""Evaluate one run folder (trained RL run or baseline) on chosen scenarios and write result files."""
# ===== EDIT HERE =====
DATASET = "mnist"                 # mnist, fmnist, cifar10
METHOD = "joint"                  # any method name from fedrl/methods.py
VARIANT = "default"               # variant folder name
TRAIN_SEED = 0                    # training seed of the run folder
SCENARIOS = "test"                # "test" (writes result JSONs) or "validation" (writes validation.json)
A_FILTER = None                   # e.g. [0.2] to keep only these skew values (test scenarios)
DRIFT_FILTER = None               # e.g. ["abrupt"]
SEED_FILTER = None                # e.g. [1000, 1001]
EVAL_OVERRIDES = []               # evaluation-only overrides, e.g. ["fl.N=50", "fl.k=10"]
TAG = ""                          # result-file tag (use "N50" together with the N=50 overrides)
FORCE = False                     # True: recompute existing results
CONFIG = "configs/default.yaml"
OVERRIDES = []                    # run-defining overrides (must match the training run, e.g. ["ppo.episodes=400"])
# ===== END EDIT =====

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedrl.config import load_config
from fedrl.data.scenario_sets import test_scenarios
from fedrl.experiments import Job, job_run_dir
from fedrl.runner import Context, execute_job
from fedrl.utils.logging_utils import get_logger

logger = get_logger("evaluate")


def main(a) -> int:
    cfg = load_config(a.config, list(a.overrides) + [f"run.dataset={a.dataset}"])
    ov = (f"run.dataset={a.dataset}",) + tuple(a.overrides)
    ctx = Context()
    base = Job("manual", a.dataset, a.method, a.variant, a.train_seed, None, None, "tune", "core", ov,
               tuple(a.eval_overrides), None, a.tag)
    print(f"run folder: {job_run_dir(base, cfg)}")
    if a.scenarios == "validation":
        print(execute_job(base, cfg, ctx, force=a.force))
        return 0
    specs = test_scenarios(cfg)
    if a.a_filter:
        specs = [s for s in specs if any(abs(s.a - x) < 1e-9 for x in a.a_filter)]
    if a.drift_filter:
        specs = [s for s in specs if s.drift in a.drift_filter]
    if a.seed_filter:
        specs = [s for s in specs if s.seed in a.seed_filter]
    n_done = 0
    for sp in specs:
        job = Job("manual", a.dataset, a.method, a.variant, a.train_seed, sp, sp.seed, "eval", "core", ov,
                  tuple(a.eval_overrides), None, a.tag)
        n_done += execute_job(job, cfg, ctx, force=a.force) == "done"
    print(f"{len(specs)} scenarios, {n_done} computed, {len(specs) - n_done} skipped (already present)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a run folder (all flags optional; defaults = EDIT HERE).")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    ap.add_argument("--scenarios", default=SCENARIOS, choices=["test", "validation"])
    ap.add_argument("--a-filter", type=float, nargs="*", default=A_FILTER)
    ap.add_argument("--drift-filter", nargs="*", default=DRIFT_FILTER)
    ap.add_argument("--seed-filter", type=int, nargs="*", default=SEED_FILTER)
    ap.add_argument("--eval-overrides", nargs="*", default=list(EVAL_OVERRIDES))
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--force", action="store_true", default=FORCE)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--set", action="append", default=list(OVERRIDES), dest="overrides")
    sys.exit(main(ap.parse_args()))
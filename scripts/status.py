"""List every job as done / partial / not_started with a summary per experiment."""
# ===== EDIT HERE =====
EXPERIMENT = "all"                # pilot, E1, ..., E7, or "all"
TIER = None                       # None/"all", "core" or "extended"
SHARD = None                      # None or "i/n"
SHOW_JOBS = True                  # False: only the per-experiment summary
CONFIG = "configs/default.yaml"
OVERRIDES = []                    # extra "key=value" overrides
# ===== END EDIT =====

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedrl.config import load_config
from fedrl.experiments import format_job, job_status, jobs_for, shard_jobs
from fedrl.experiments import EXPERIMENT_TIERS, experiments_in_tier


def main(experiment: str, tier: str | None, shard: str | None, show_jobs: bool, config: str, overrides: list[str]) -> int:
    cfg = load_config(config, overrides)
    exps = experiments_in_tier(tier) if experiment == "all" else [experiment]
    for e in exps:
        if e not in EXPERIMENT_TIERS:
            raise SystemExit(f"unknown experiment '{e}'")
    sh = tuple(int(x) for x in shard.split("/")) if shard else None
    grand: Counter = Counter()
    for e in exps:
        jobs = shard_jobs(jobs_for(e, cfg), sh)
        tally: Counter = Counter()
        print(f"\n===== {e} ({EXPERIMENT_TIERS[e]}) =====")
        for j in jobs:
            s = job_status(j, cfg)
            tally[(j.kind, s)] += 1
            if show_jobs:
                print(f"  {s:12s} {format_job(j)}")
        print(f"  -- summary {e}: " + ", ".join(
            f"{k}: " + "/".join(f"{tally[(k, s)]} {s}" for s in ("done", "partial", "not_started"))
            for k in ("train", "tune", "eval") if any(tally[(k, s)] for s in ("done", "partial", "not_started"))) or "no jobs")
        grand.update(tally)
    print("\nOVERALL: " + ", ".join(f"{s}={sum(v for (k, st), v in grand.items() if st == s)}"
                                    for s in ("done", "partial", "not_started")))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Job status (all flags optional; defaults = EDIT HERE).")
    ap.add_argument("--experiment", default=EXPERIMENT)
    ap.add_argument("--tier", default=TIER)
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--no-jobs", action="store_true", default=not SHOW_JOBS)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--set", action="append", default=list(OVERRIDES), dest="overrides")
    a = ap.parse_args()
    sys.exit(main(a.experiment, a.tier, a.shard, not a.no_jobs, a.config, a.overrides))
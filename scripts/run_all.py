"""Run experiments: train jobs, tune jobs, then eval jobs. Automatic resume; finished jobs are skipped."""
# ===== EDIT HERE =====
EXPERIMENT = "pilot"              # pilot, E1, E2, E3, E4, E5, E2b, E6, E7, or "all" (all experiments of TIER)
TIER = None                       # None/"all", "core" or "extended" (used when EXPERIMENT == "all")
DRY_RUN = False                   # True: print every job and counts, run nothing
SHARD = None                      # None or "i/n": this machine runs shard i of n (run index mod n)
MAX_PARALLEL = 1                  # number of subprocesses, each with one CPU thread
CONFIG = "configs/default.yaml"   # base config file
OVERRIDES = []                    # extra "key=value" overrides, e.g. ["run.runs_dir=./runs"]
FORCE = False                     # True: recompute finished jobs
# ===== END EDIT =====

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedrl.config import load_config
from fedrl.experiments import (EXPERIMENT_TIERS, count_jobs, dedupe, experiments_in_tier, format_job,
                               job_status, jobs_for, shard_jobs)
from fedrl.runner import run_jobs, spawn_workers


def parse_shard(text: str | None) -> tuple[int, int] | None:
    """'i/n' -> (i, n)."""
    if not text:
        return None
    i, n = text.split("/")
    return int(i), int(n)


def select_experiments(experiment: str, tier: str | None) -> list[str]:
    """Experiment names to run."""
    if experiment == "all":
        return experiments_in_tier(tier)
    if experiment not in EXPERIMENT_TIERS:
        raise SystemExit(f"unknown experiment '{experiment}'; choose from {sorted(EXPERIMENT_TIERS)} or 'all'")
    if tier not in (None, "all") and EXPERIMENT_TIERS[experiment] != tier:
        raise SystemExit(f"{experiment} belongs to tier '{EXPERIMENT_TIERS[experiment]}', not '{tier}'")
    return [experiment]


def main(experiment: str, tier: str | None, dry_run: bool, shard_text: str | None, max_parallel: int,
         config: str, overrides: list[str], force: bool) -> int:
    cfg = load_config(config, overrides)
    exps = select_experiments(experiment, tier)
    per_exp = {e: jobs_for(e, cfg) for e in exps}
    jobs = dedupe([j for e in exps for j in per_exp[e]])
    jobs = shard_jobs(jobs, parse_shard(shard_text))
    worker = os.environ.get("FEDRL_WORKER")
    if worker:
        jobs = shard_jobs(jobs, parse_shard(worker))
    if dry_run:
        for j in jobs:
            print(f"{job_status(j, cfg):12s} {format_job(j)}")
        print("\nJobs per experiment (before de-duplication and sharding):")
        for e in exps:
            c = count_jobs(per_exp[e])
            print(f"  {e:6s} train={c['train']:4d} tune={c['tune']:3d} eval={c['eval']:5d} total={c['total']:5d}")
        c = count_jobs(jobs)
        print(f"TOTAL to run (after de-dup/sharding): train={c['train']} tune={c['tune']} eval={c['eval']} total={c['total']}")
        return 0
    if max_parallel > 1 and not worker:
        args = ["--experiment", experiment, "--tier", str(tier or "all"), "--config", config, "--max-parallel", "1"]
        if shard_text:
            args += ["--shard", shard_text]
        for s in overrides:
            args += ["--set", s]
        if force:
            args.append("--force")
        codes = spawn_workers(str(Path(__file__).resolve()), args, max_parallel)
        print(f"worker exit codes: {codes}")
        return 0 if all(c == 0 for c in codes) else 1
    stats = run_jobs(jobs, cfg, force=force)
    print(f"executed={stats['done']} skipped={stats['skipped']} failed={len(stats['failed'])}")
    for name, err in stats["failed"]:
        print(f"  FAILED {name}: {err}")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run experiments (all flags optional; defaults = EDIT HERE).")
    ap.add_argument("--experiment", default=EXPERIMENT)
    ap.add_argument("--tier", default=TIER)
    ap.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--max-parallel", type=int, default=MAX_PARALLEL)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--set", action="append", default=list(OVERRIDES), dest="overrides")
    ap.add_argument("--force", action="store_true", default=FORCE)
    a = ap.parse_args()
    sys.exit(main(a.experiment, a.tier, a.dry_run, a.shard, a.max_parallel, a.config, a.overrides, a.force))
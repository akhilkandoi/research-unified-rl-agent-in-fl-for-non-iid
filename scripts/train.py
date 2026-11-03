"""Train one run (or all train jobs of an experiment). Run: python scripts/train.py"""
# ===== EDIT HERE =====
CONFIG = "configs/default.yaml"   # base config file (configs/smoke.yaml for a quick check)
OVERRIDES = []                    # e.g. ["fl.k=3", "ppo.episodes=100"]
DATASET = None                    # None = take run.dataset from config ("mnist", "fmnist", "cifar10")
METHOD = "joint"                  # any name from fedrl/methods.py
VARIANT = "default"               # "default", "wcomm0", "wfair0", "wcomm0_wfair0", "upd4", ...
SEED = 0                          # training seed
EPISODES = None                   # None = config value; raising it extends a finished run
SHARD = None                      # None or "i/n" (only used with EXPERIMENT)
EXPERIMENT = None                 # e.g. "E1": train every train-job of that experiment (sharded)
# ===== END EDIT =====
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedrl.config import load_config  # noqa: E402
from fedrl.rl.trainer import train  # noqa: E402
from fedrl.runs.locks import parse_shard, shard_jobs  # noqa: E402


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optional overrides of the EDIT HERE variables.")
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--set", action="append", default=[], dest="sets", help="key=value (repeatable)")
    p.add_argument("--method", default=METHOD)
    p.add_argument("--variant", default=VARIANT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--shard", default=SHARD)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--experiment", default=EXPERIMENT)
    return p.parse_args()


def main() -> None:
    a = _args()
    overrides = list(OVERRIDES) + list(a.sets)
    if a.dataset:
        overrides.append(f"run.dataset={a.dataset}")
    if a.episodes is not None:
        overrides.append(f"ppo.episodes={a.episodes}")
    cfg = load_config(a.config, overrides)
    logging.basicConfig(level=getattr(logging, str(cfg.logging.level), logging.INFO))
    shard = parse_shard(a.shard)
    if a.experiment:
        from fedrl.experiments import jobs_for
        jobs = [j for j in jobs_for(a.experiment, cfg) if j.kind == "train"]
        if shard:
            jobs = shard_jobs(jobs, *shard)
        for j in jobs:
            jcfg = load_config(a.config, overrides + [f"run.dataset={j.dataset}"])
            rd = train(jcfg, j.method, j.variant, j.train_seed, shard)
            print(f"[{j.method}/{j.variant}/seed{j.train_seed}] run dir: {rd}")
        return
    rd = train(cfg, a.method, a.variant, a.seed, shard)
    print(f"Run directory: {rd}")
    print(f"Logs: {rd/'train_log.csv'}, {rd/'val_log.csv'}, {rd/'selection_counts.csv'}")
    print(f"Checkpoints: {rd/'checkpoints'} (rerun the same command to resume)")


if __name__ == "__main__":
    main()
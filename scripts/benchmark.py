"""Timing benchmark for FL rounds (CPU/GPU, thread counts) with a clearly labeled projection.

Run: python scripts/benchmark.py   (edit the block below; flags are optional overrides)
"""
# ===== EDIT HERE =====
DATASET = "mnist"                       # mnist | fmnist | cifar10
CONFIG = "configs/default.yaml"         # base config
OVERRIDES: list = []                    # extra "a.b=value" overrides
DEVICES = ["cpu", "cuda"]               # cuda is skipped if unavailable
THREADS = [1, 2, 4]                     # torch CPU thread counts to try
SIZES = [(20, 5, "main"), (10, 3, "pilot")]   # (N, k, tag)
WARMUP_ROUNDS = 3                       # unmeasured rounds per configuration
MEASURE_ROUNDS = 12                     # measured rounds per configuration (mean/std over these)
SEED = 0                                # scenario seed
SKEW_A = 0.3                            # Dirichlet skew of the benchmark scenario
DRIFT = "none"                          # none | abrupt | gradual
OUT_DIR = "benchmark_results"           # JSON and CSV go here
ALLOW_SYNTHETIC_DATA = True             # time on random data of the right shape if the download fails
PARALLEL_PROCESSES = [1, 2, 3]          # for the projection table
DRY_RUN = False                         # True: 1 warm-up and 2 measured rounds (smoke check)
# ===== END EDIT =====

import argparse
import csv
import gc
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedrl.config import load_config  # noqa: E402
from fedrl.data.datasets import DatasetBundle, load_dataset  # noqa: E402
from fedrl.data.pools import build_pools  # noqa: E402
from fedrl.data.scenario import Scenario  # noqa: E402
from fedrl.data.scenario_sets import ScenarioSpec  # noqa: E402
from fedrl.fl.fl_round import FLWorld, make_action  # noqa: E402
from fedrl.fl.resources import eligible_mask  # noqa: E402
from fedrl.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("benchmark")

COMPONENTS = ["probe", "train", "aggregate", "pers_eval", "global_eval", "other", "total"]
# Core RL trainings from MASTER 2.14: E1 3x3, E2 5x2, E3 4x1, E4 3x3 (plus 5 tuning/bandit runs).
RL_RUNS = {"E1": 9, "E2": 10, "E3": 4, "E4": 9}
TUNING_RUNS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FL round timing benchmark (all flags optional).")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--warmup", type=int, default=WARMUP_ROUNDS)
    p.add_argument("--rounds", type=int, default=MEASURE_ROUNDS)
    p.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    return p.parse_args()


def synthetic_bundle(dataset: str, device: torch.device) -> DatasetBundle:
    """Random data with the real shapes (timing does not depend on pixel values)."""
    shape, n_train = ((3, 32, 32), 50000) if dataset == "cifar10" else ((1, 28, 28), 60000)
    g = torch.Generator().manual_seed(0)
    mk = lambda n: (torch.randn(n, *shape, generator=g).to(device),
                    torch.randint(0, 10, (n,), generator=g).to(device))
    tx, ty = mk(n_train)
    ex, ey = mk(10000)
    return DatasetBundle(name=dataset, num_classes=10, train_x=tx, train_y=ty,
                         test_x=ex, test_y=ey, device=device)


def get_bundle(dataset: str, cfg, device: torch.device):
    try:
        return load_dataset(dataset, cfg.run.data_dir, device), "real"
    except Exception as exc:  # network / torchvision problems
        if not ALLOW_SYNTHETIC_DATA:
            raise
        log.warning("Dataset load failed (%s). Using SYNTHETIC data of the same shape; "
                    "timings remain valid, accuracies are meaningless.", exc)
        return synthetic_bundle(dataset, device), "synthetic"


def bench_one(cfg, bundle, pools, device, warmup: int, measure: int) -> dict:
    """Time `measure` rounds at median settings with k random eligible clients."""
    spec = ScenarioSpec(a=SKEW_A, drift=DRIFT, seed=SEED)
    world = FLWorld(Scenario(spec, bundle, pools, cfg), bundle, pools, cfg, device)
    world.profile = True
    rng = np.random.default_rng(SEED)
    N, k, T = world.N, int(cfg.fl.k), int(cfg.fl.T)
    t, recs = 1, []
    for r in range(warmup + measure):
        if t > T:
            world.reset()
            t = 1
        elig = np.flatnonzero(eligible_mask(world.battery, world.n, cfg).cpu().numpy())
        pick = rng.choice(elig, size=min(k, len(elig)), replace=False) if len(elig) else []
        res = world.run_round(t, make_action(cfg, N, [int(i) for i in pick]))
        t += 1
        if r >= warmup:
            recs.append(res.timings)
    del world
    return {c: [rec[c] for rec in recs] for c in COMPONENTS}


def fmt(seconds: float) -> str:
    return f"{seconds:,.0f} s ({seconds / 3600:.2f} h)"


def projection(rows: list, cfg, threads_ref: int) -> list:
    """Projection from measured seconds/round (threads=threads_ref, 'main' size). Not a measurement."""
    out = []
    episodes, T = int(cfg.ppo.episodes), int(cfg.fl.T)
    n_val = len(cfg.scenarios.val.a) * len(cfg.scenarios.val.drift) * len(cfg.scenarios.val.seeds)
    val_rounds = (episodes // int(cfg.ppo.validate_every)) * n_val * T
    train_rounds = episodes * T
    budget = int(cfg.tuning.budget_rounds) if cfg.tuning.budget_rounds else train_rounds
    print("\n" + "=" * 100)
    print("PROJECTION (derived from measured seconds/round; NOT a measurement)")
    print(f"episodes={episodes}, T={T}, validation rounds per run={val_rounds}, "
          f"RL trainings={sum(RL_RUNS.values())} ({RL_RUNS}), tuning/bandit runs={TUNING_RUNS} "
          f"x {budget} rounds")
    print("Excludes PPO update time and final-evaluation time (not measured here). "
          "Parallel columns assume ideal scaling: p processes each at the measured single-process speed; "
          "contention (memory bandwidth, shared GPU) is not measured.")
    for dev in sorted({r["device"] for r in rows}):
        sel = [r for r in rows if r["device"] == dev and r["tag"] == "main" and r["component"] == "total"
               and r["threads"] == threads_ref]
        if not sel:
            continue
        spr = sel[0]["mean_s"]
        one = train_rounds * spr
        one_val = (train_rounds + val_rounds) * spr
        core = sum(RL_RUNS.values()) * one_val
        tune = TUNING_RUNS * budget * spr
        total = core + tune
        print(f"\n[{dev}, threads={threads_ref}] seconds/round = {spr:.4f}")
        print(f"  one RL training, FL rounds only      : {fmt(one)}")
        print(f"  one RL training incl. validation     : {fmt(one_val)}")
        print(f"  core RL trainings (all, incl. val)   : {fmt(core)}")
        print(f"  tuning/bandit runs                   : {fmt(tune)}")
        print(f"  TOTAL sequential                     : {fmt(total)}")
        for p in PARALLEL_PROCESSES:
            print(f"  wall-clock with {p} parallel process(es): {fmt(total / p)}")
            out.append({"device": dev, "threads": threads_ref, "parallel": p, "seconds_per_round": spr,
                        "total_sequential_s": total, "wall_s": total / p})
        pil = [r for r in rows if r["device"] == dev and r["tag"] == "pilot" and r["component"] == "total"
               and r["threads"] == threads_ref]
        if pil:
            print(f"  one pilot-size training (300 ep x 20 rounds, FL rounds only): "
                  f"{fmt(300 * 20 * pil[0]['mean_s'])}")
    return out


def main() -> None:
    args = parse_args()
    warmup, measure = (1, 2) if args.dry_run else (args.warmup, args.rounds)
    threads_list = [THREADS[0]] if args.dry_run else THREADS
    base = load_config(args.config, overrides=list(OVERRIDES) + [f"run.dataset={args.dataset}"])
    rows, meta_data = [], {}
    for dev_name in DEVICES:
        if dev_name == "cuda" and not torch.cuda.is_available():
            log.info("CUDA not available: skipping GPU.")
            continue
        device = torch.device(dev_name)
        bundle, source = get_bundle(args.dataset, base, device)
        meta_data[dev_name] = source
        for threads in threads_list:
            torch.set_num_threads(threads)
            for (N, k, tag) in SIZES:
                cfg = load_config(args.config, overrides=list(OVERRIDES) +
                                  [f"run.dataset={args.dataset}", f"fl.N={N}", f"fl.k={k}"])
                pools = build_pools(bundle, cfg)
                log.info("device=%s threads=%d N=%d k=%d (%s)", dev_name, threads, N, k, tag)
                res = bench_one(cfg, bundle, pools, device, warmup, measure)
                for comp, vals in res.items():
                    arr = np.asarray(vals)
                    rows.append({"device": dev_name, "threads": threads, "tag": tag, "N": N, "k": k,
                                 "component": comp, "mean_s": float(arr.mean()),
                                 "std_s": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                                 "n_rounds": int(len(arr))})
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del bundle
    if not rows:
        raise SystemExit("No measurements were taken.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "data_source": meta_data, "warmup": warmup, "measure": measure,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    threads_ref = threads_list[0]
    proj = projection(rows, base, threads_ref)
    (out / "benchmark.json").write_text(json.dumps({"meta": meta, "rows": rows, "projection": proj}, indent=2))
    with open(out / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nMEASURED seconds per round (mean +- std):")
    for r in rows:
        if r["component"] in ("total", "train"):
            print(f"  {r['device']:4s} thr={r['threads']} {r['tag']:5s} N={r['N']:2d} k={r['k']} "
                  f"{r['component']:6s} {r['mean_s']:.4f} +- {r['std_s']:.4f}")
    print(f"\nWrote {out / 'benchmark.json'} and {out / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
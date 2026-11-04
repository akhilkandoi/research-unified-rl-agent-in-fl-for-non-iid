"""Reward sanity check: fixed-action episodes on one scenario. Run before long training."""
# ===== EDIT HERE =====
CONFIG = "configs/default.yaml"   # config file (use configs/smoke.yaml for a quick wiring test)
OVERRIDES = []                    # e.g. ["fl.T=10"]
DATASET = "mnist"                 # mnist | fmnist | cifar10
SEED = 0                          # scenario seed and selection-RNG seed
A_SKEW = 0.3                      # Dirichlet skew of the checked scenario
DRIFT = "abrupt"                  # none | abrupt | gradual
BAD_LR = 20.0                     # deliberately bad learning rate
PERS_MODEST_MAX = 5.0             # |sum r_pers| per episode still called "modest" (reward units)
COST_TOL = 0.35                   # random policy normalized cost must be within 1 +/- COST_TOL
# ===== END EDIT =====

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from fedrl.config import load_config, resolve_device
from fedrl.data.datasets import load_dataset
from fedrl.data.pools import build_pools
from fedrl.data.scenario_sets import ScenarioSpec
from fedrl.env.fl_env import FLEnv
from fedrl.utils.logging_utils import get_logger

log = get_logger("reward_check")
SUM_KEYS = ["r_global", "r_pers", "r_fair", "final_bonus"]


def make_action(obs, env, rng, cfg, lr_value=None, alpha_value=None, randomize=False):
    """Random eligible selection plus fixed (or fully random) settings."""
    N = env.N
    elig = obs["eligible"].nonzero().flatten().cpu().numpy()
    k_eff = min(env.k, len(elig))
    sel = rng.choice(elig, size=k_eff, replace=False)
    act = {"selected": torch.as_tensor(sel, dtype=torch.long)}
    if randomize:
        for key, menu in (("lr_idx", cfg.menus.lr), ("epochs_idx", cfg.menus.epochs),
                          ("batch_idx", cfg.menus.batch), ("alpha_idx", cfg.menus.alpha)):
            act[key] = torch.as_tensor(rng.integers(0, len(menu), size=N), dtype=torch.long)
    if lr_value is not None:
        act["lr_value"] = torch.full((N,), float(lr_value))
    if alpha_value is not None:
        act["alpha_value"] = torch.full((N,), float(alpha_value))
    return act


def run_episode(env, spec, cfg, name, **kw):
    rng = np.random.default_rng(SEED)
    obs = env.reset(spec)
    done = False
    sums = {k: 0.0 for k in SUM_KEYS}
    comm, comp, gain_all, alpha, rounds = 0.0, 0.0, 0.0, 0.0, 0
    ret, acc = 0.0, float("nan")
    while not done:
        obs, r, done, info = env.step(make_action(obs, env, rng, cfg, **kw))
        ret += r
        for k in SUM_KEYS:
            sums[k] += info[k]
        comm += info["comm_norm"]; comp += info["comp_norm"]
        gain_all += info["pers_gain_all"]; alpha += info["mean_alpha"]
        acc = info["global_val_acc"]; rounds += 1
    return {"name": name, "return": ret, **sums, "comm_norm": comm / rounds, "comp_norm": comp / rounds,
            "final_acc": acc, "pers_gain_all": gain_all / rounds, "mean_alpha": alpha / rounds}


def fmt(x):
    return f"{x:9.3f}" if isinstance(x, float) else f"{x!s:>9}"


def print_table(rows):
    cols = ["return", "r_global", "r_pers", "r_fair", "final_bonus", "comm_norm", "comp_norm",
            "final_acc", "pers_gain_all", "mean_alpha"]
    print(f"{'episode':<22}" + "".join(f"{c:>14}" for c in cols))
    for r in rows:
        print(f"{r['name']:<22}" + "".join(f"{r[c]:>14.4f}" for c in cols))


def verdict(rows):
    by = {r["name"]: r for r in rows}
    ok = math.isfinite
    print("\nSANITY VERDICT (from measured numbers only)")
    a, b, c = by["lr=0.01"]["return"], by["lr=0.001"]["return"], by[f"lr={BAD_LR:g}"]["return"]
    good = ok(a) and ((not ok(b)) or a > b) and ((not ok(c)) or a > c)
    print(f"[{'OK' if good else 'CHECK'}] return lr0.01={a:.3f}, lr0.001={b:.3f}, lr{BAD_LR:g}={c:.3f} "
          "(expect lr0.01 best)")
    p = by["alpha=0.5"]["r_pers"]
    print(f"[{'OK' if ok(p) and abs(p) <= PERS_MODEST_MAX else 'CHECK'}] alpha0.5 sum r_pers={p:.3f}; "
          f"alpha0.1={by['alpha=0.1']['r_pers']:.3f}, alpha0.9={by['alpha=0.9']['r_pers']:.3f} "
          f"(expect |alpha0.5| <= {PERS_MODEST_MAX})")
    rr = by["random"]
    cost = 0.5 * (rr["comm_norm"] + rr["comp_norm"])
    print(f"[{'OK' if ok(cost) and abs(cost - 1.0) <= COST_TOL else 'CHECK'}] random policy mean "
          f"normalized cost={cost:.3f} (comm {rr['comm_norm']:.3f}, comp {rr['comp_norm']:.3f}); expect ~1")


def main():
    global SEED
    ap = argparse.ArgumentParser(description="Reward sanity check (all flags optional)")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--set", dest="overrides", action="append", default=None)
    args = ap.parse_args()
    SEED = args.seed
    overrides = list(args.overrides if args.overrides is not None else OVERRIDES)
    overrides.append(f"run.dataset={args.dataset}")
    cfg = load_config(args.config, overrides)
    torch.set_num_threads(int(cfg.run.num_threads))
    device = resolve_device(cfg)
    bundle = load_dataset(cfg.run.dataset, cfg.run.data_dir, device)
    pools = build_pools(bundle, cfg)
    env = FLEnv(cfg, bundle, pools, device)
    spec = ScenarioSpec(a=A_SKEW, drift=DRIFT, seed=SEED)
    plan = [
        ("lr=0.01", {"lr_value": 0.01, "alpha_value": 0.5}),
        ("lr=0.001", {"lr_value": 0.001, "alpha_value": 0.5}),
        (f"lr={BAD_LR:g}", {"lr_value": BAD_LR, "alpha_value": 0.5}),
        ("alpha=0.5", {"lr_value": 0.01, "alpha_value": 0.5}),
        ("alpha=0.1", {"lr_value": 0.01, "alpha_value": 0.1}),
        ("alpha=0.9", {"lr_value": 0.01, "alpha_value": 0.9}),
        ("random", {"randomize": True}),
    ]
    rows = []
    for name, kw in plan:
        log.info("episode %s", name)
        rows.append(run_episode(env, spec, cfg, name, **kw))
    print(f"\nScenario: a={A_SKEW} drift={DRIFT} seed={SEED} dataset={cfg.run.dataset} "
          f"(per-episode sums; costs are per-round means)\n")
    print_table(rows)
    verdict(rows)


if __name__ == "__main__":
    main()
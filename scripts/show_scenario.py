"""Print client sizes, class mixes, resources and a before/after drift summary for one scenario.

Run: python scripts/show_scenario.py   (edit the block below; flags are optional overrides)
"""
# ===== EDIT HERE =====
DATASET = "mnist"                    # mnist | fmnist | cifar10
A = 0.2                              # Dirichlet skew (small = strong skew)
DRIFT = "abrupt"                     # none | abrupt | gradual
SEED = 1000                          # scenario seed
CONFIG = "configs/default.yaml"      # config file
OVERRIDES: list[str] = []            # e.g. ["fl.N=10"]
# ===== END EDIT =====

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedrl.config import load_config, resolve_device  # noqa: E402
from fedrl.data.datasets import load_dataset  # noqa: E402
from fedrl.data.pools import build_pools  # noqa: E402
from fedrl.data.scenario import Scenario  # noqa: E402
from fedrl.data.scenario_sets import ScenarioSpec  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--a", type=float, default=A)
    ap.add_argument("--drift", default=DRIFT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--set", action="append", default=list(OVERRIDES), dest="overrides")
    return ap.parse_args()


def mix_str(p: np.ndarray) -> str:
    return " ".join(f"{x:4.2f}" for x in p)


def stats(name: str, v: np.ndarray) -> str:
    return (f"{name:<14} mean={v.mean():6.3f} median={np.median(v):6.3f} "
            f"min={v.min():6.3f} max={v.max():6.3f}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [f"run.dataset={args.dataset}"] + list(args.overrides))
    device = resolve_device(cfg)
    bundle = load_dataset(cfg.run.dataset, cfg.run.data_dir, device)
    pools = build_pools(bundle, cfg)
    sc = Scenario(ScenarioSpec(args.a, args.drift, args.seed), bundle, pools, cfg)
    ytr = bundle.train_y.cpu().numpy()
    C = bundle.num_classes

    print(f"\nScenario: dataset={cfg.run.dataset} a={args.a} drift={args.drift} seed={args.seed} "
          f"N={sc.N} init_seed={sc.init_seed}")
    print("\nClients (proportions per class 0..%d):" % (C - 1))
    print(f"{'id':>3} {'n':>5} {'val':>4} {'pval':>4} {'speed':>6} {'bw':>6} {'bat0':>5}  mix")
    for i, c in enumerate(sc.clients):
        print(f"{i:>3} {c.n:>5} {len(c.val_idx):>4} {len(c.personal_val_idx):>4} {c.compute_speed:>6.2f} "
              f"{c.bandwidth:>6.2f} {c.battery0:>5.2f}  {mix_str(c.proportions)}")
    sizes = np.array([c.n for c in sc.clients], dtype=float)
    print("\nSummary:")
    print(stats("size", sizes))
    print(stats("compute_speed", np.array([c.compute_speed for c in sc.clients])))
    print(stats("bandwidth", np.array([c.bandwidth for c in sc.clients])))
    print(stats("battery0", np.array([c.battery0 for c in sc.clients])))
    emp = np.stack([np.bincount(ytr[c.train_idx.cpu().numpy()], minlength=C) / max(c.n, 1) for c in sc.clients])
    print(f"mean L1(empirical train mix, proportions) = "
          f"{np.abs(emp - sc.proportions_matrix()).sum(1).mean():.3f}")

    before = sc.proportions_matrix().copy()
    changed_rounds = [t for t in range(1, int(cfg.fl.T) + 1) if sc.apply_drift(t)]
    after = sc.proportions_matrix()
    print(f"\nDrift: kind={args.drift}, rounds with data change: {changed_rounds or 'none'}")
    if changed_rounds:
        shift = np.abs(after - before).sum(1)
        print(f"L1 shift per client: mean={shift.mean():.3f} min={shift.min():.3f} max={shift.max():.3f}")
        print(f"{'id':>3} {'top before':>10} {'top after':>10} {'L1':>5}")
        for i in range(sc.N):
            print(f"{i:>3} {int(before[i].argmax()):>10} {int(after[i].argmax()):>10} {shift[i]:>5.2f}")
    p0 = sc.personal_test_idx(0)
    cnt = np.bincount(bundle.test_y[p0].cpu().numpy(), minlength=C)
    print(f"\nPersonal test set of client 0 ({len(p0)} samples), class counts: {cnt.tolist()}")


if __name__ == "__main__":
    main()
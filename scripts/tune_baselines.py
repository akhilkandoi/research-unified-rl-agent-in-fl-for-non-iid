"""Tune tuned_settings and tuned_alpha on VALIDATION scenarios (budget capped)."""
# ===== EDIT HERE =====
CONFIG = "configs/default.yaml"   # config file (default.yaml is always loaded first)
DATASET = "mnist"                 # mnist | fmnist | cifar10
OVERRIDES: list[str] = []         # e.g. ["fl.k=3", "tuning.budget_rounds=20000"]
WHAT = ("tuned_settings", "tuned_alpha")  # which tuners to run
# ===== END EDIT =====

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedrl.config import load_config, validate  # noqa: E402
from fedrl.controllers.fixed import tune_alpha, tune_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--set", action="append", default=None, dest="overrides")
    parser.add_argument("--only", choices=["tuned_settings", "tuned_alpha"], default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = [f"run.dataset={args.dataset}"] + list(args.overrides if args.overrides is not None
                                                       else OVERRIDES)
    cfg = load_config(args.config, overrides)
    validate(cfg)
    todo = (args.only,) if args.only else tuple(WHAT)
    for name in todo:
        result = tune_settings(cfg) if name == "tuned_settings" else tune_alpha(cfg)
        print(f"[{name}] budget used: {result['rounds_used']} / {result['budget_cap']} FL rounds")
        print(f"[{name}] chosen: {result['chosen_values']}  "
              f"(mean validation return {result['score']:.4f})")
        print(f"[{name}] wrote {result['path']}")


if __name__ == "__main__":
    main()
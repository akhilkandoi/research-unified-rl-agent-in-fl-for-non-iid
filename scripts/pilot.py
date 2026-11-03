"""Pilot go/no-go (run this first). Trains/loads pilot runs, then reports plateau and D12 criteria.

Uses validation returns only. Resume is automatic. Episodes are never added automatically.
"""
# ===== EDIT HERE =====
CONFIG = "configs/default.yaml"   # base config (jobs additionally use configs/pilot.yaml)
OVERRIDES = []                    # extra "key=value" overrides for the base config (e.g. ["run.runs_dir=./runs"])
DRY_RUN = False                   # True: only list the pilot jobs and exit
SKIP_TRAINING = False             # True: do not train, only analyse existing runs
OUT_DIR = "pilot_out"             # where learning-curve plots are written
PLATEAU_MIN_TAIL = 0.2            # a curve "plateaus" only if a flat tail of >= this fraction of points exists
# ===== END EDIT =====

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fedrl.config import load_config
from fedrl.data.scenario_sets import validation_scenarios
from fedrl.eval.evaluate_policy import collect_diagnostics, load_validation_result, run_episode
from fedrl.eval.pilot import evaluate_pilot_criteria, format_criteria, sweep_plateau
from fedrl.experiments import PILOT_SEEDS, PILOT_UPDATE_SWEEP, count_jobs, format_job, job_config, job_run_dir, jobs_for, job_status
from fedrl.runner import Context, build_run_controller, run_jobs
from fedrl.utils.logging_utils import get_logger
from fedrl.eval.logs import read_alpha_curve, read_val_curve
from fedrl.methods import is_rl

logger = get_logger("pilot")


def _curves(jobs, cfg, variant: str, method: str):
    """(episodes [P], curves [S,P], alphas list) of validation returns aligned across seeds."""
    series, alphas = [], []
    for j in jobs:
        if j.kind != "train" or j.method != method or j.variant != variant:
            continue
        rd = job_run_dir(j, cfg)
        v = read_val_curve(rd)
        if v is None:
            continue
        series.append(pd.Series(v[1], index=v[0], name=f"seed{j.train_seed}"))
        a = read_alpha_curve(rd)
        if a is not None:
            alphas.append(a)
    if not series:
        return None
    mat = pd.concat(series, axis=1, join="inner")
    return mat.index.to_numpy(), mat.to_numpy().T, alphas


def _selection_counts(job, cfg, ctx: Context) -> list[list[int]]:
    """Per-episode pick counts of the trained policy (deterministic) on the VALIDATION scenarios."""
    ctrl, cfg_a, _ = build_run_controller(job, cfg)
    env = ctx.env(cfg_a, ("pilot", job.variant, job.train_seed))
    out = []
    for spec in validation_scenarios(cfg_a):
        ep = run_episode(ctrl, env, spec, spec.seed, cfg_a)
        out.append(collect_diagnostics(ep["infos"], ep["rewards"], env.world.N)["selection_counts"])
    return out


def _plot(curves: dict, baselines: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, seed in enumerate(PILOT_SEEDS):
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, (eps, arr, _) in curves.items():
            if s < arr.shape[0]:
                ax.plot(eps, arr[s], marker="o", label=label)
        for name, val in baselines.items():
            ax.axhline(val, ls="--", lw=1, label=f"{name} (val)")
        ax.set_xlabel("episode")
        ax.set_ylabel("validation return")
        ax.set_title(f"Pilot learning curves, training seed {seed}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"pilot_learning_curve_seed{seed}.png", dpi=130)
        plt.close(fig)


def main(config: str, overrides: list[str], dry_run: bool, skip_training: bool, out_dir: str) -> int:
    cfg = load_config(config, overrides)
    jobs = jobs_for("pilot", cfg)
    c = count_jobs(jobs)
    print(f"pilot jobs: {c['train']} train, {c['tune']} tune, total {c['total']}")
    if dry_run:
        for j in jobs:
            print(f"  {job_status(j, cfg):12s} {format_job(j)}")
        return 0
    if not skip_training:
        stats = run_jobs(jobs, cfg)
        print(f"executed={stats['done']} skipped={stats['skipped']} failed={len(stats['failed'])}")
    for j in jobs:
        if j.kind == "train" and job_status(j, cfg) != "done":
            print(f"WARNING: {format_job(j)} is not finished; analysing partial run")

    curves = {}
    for u in PILOT_UPDATE_SWEEP:
        r = _curves(jobs, cfg, f"upd{u}", "joint")
        if r is not None:
            curves[f"joint u{u}"] = r
    st = _curves(jobs, cfg, "upd2", "stacked_shared")
    if st is not None:
        curves["stacked pair u2"] = st
    if not any(k.startswith("joint") for k in curves):
        print("No pilot validation curves found (train first).")
        return 1

    baselines = {}
    for j in jobs:
        if not is_rl(j.method):
            v = load_validation_result(job_run_dir(j, cfg))
            if v is not None:
                baselines[j.method] = float(v["mean_return"])

    print("\n== Validation return at the end of training (mean +/- std over seeds) ==")
    for label, (eps, arr, _) in curves.items():
        print(f"  {label:18s} {arr[:, -1].mean():9.3f} +/- {arr[:, -1].std():.3f}  (episode {int(eps[-1])}, {arr.shape[0]} seeds)")
    for name, val in baselines.items():
        print(f"  {name:18s} {val:9.3f}  (non-RL, validation scenarios)")

    sweep = sweep_plateau({int(k.split('u')[-1]): v[1] for k, v in curves.items() if k.startswith("joint")},
                          min_tail=PLATEAU_MIN_TAIL)
    print("\n== Update-frequency sweep (D11 plateau rule) ==")
    for u, info in sorted(sweep["per_u"].items()):
        eps = curves[f"joint u{u}"][0]
        print(f"  update_every={u}: plateau starts at episode {int(eps[info['index']])} "
              f"(SE {info['se']:.3f}, plateaus={info['plateaus']})")
    sel = sweep["selected"]
    if sel is None:
        print("  No update frequency reaches a plateau within the budget; using update_every=2 for the criteria.")
        sel = 2
    else:
        print(f"  Smallest update_every at which the curve plateaus: {sel}")

    label = f"joint u{sel}"
    eps, arr, alphas = curves[label]
    ctx = Context()
    counts = []
    for j in jobs:
        if j.method == "joint" and j.variant == f"upd{sel}":
            try:
                counts += _selection_counts(j, cfg, ctx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not evaluate selection counts for %s: %r", format_job(j), exc)
    res = evaluate_pilot_criteria(arr, alphas, counts)
    print(f"\n== D12 pilot criteria for {label} ==")
    print(format_criteria(res))
    _plot(curves, baselines, Path(out_dir))
    print(f"\nPlots written to {out_dir}/")
    return 0 if res["all_pass"] else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pilot go/no-go (all flags optional; defaults = EDIT HERE).")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--set", action="append", default=list(OVERRIDES), dest="overrides")
    ap.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    ap.add_argument("--skip-training", action="store_true", default=SKIP_TRAINING)
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    sys.exit(main(a.config, a.overrides, a.dry_run, a.skip_training, a.out_dir))
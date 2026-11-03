"""Shared job execution (train / tune / eval) used by scripts/run_all.py, pilot.py and evaluate.py."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import importlib

from tqdm import tqdm

from fedrl.config import Config, resolve_device
from fedrl.controllers import build_controller
from fedrl.data.datasets import load_dataset
from fedrl.data.pools import build_pools
from fedrl.env.fl_env import FLEnv
from fedrl.env.state import state_dim
from fedrl.eval.evaluate_policy import evaluate_and_save, evaluate_on_validation, save_validation_result
from fedrl.experiments import Job, format_job, is_trained, job_config, job_run_dir, job_status
from fedrl.methods import is_rl
from fedrl.rl.trainer import train
from fedrl.runs.checkpoint import CheckpointManager
from fedrl.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Context:
    """Per-process cache of datasets, pools and environments (dataset loaded once on device)."""

    def __init__(self) -> None:
        self._data: dict = {}
        self._envs: dict = {}

    def data(self, cfg: Config) -> tuple[Any, Any, Any]:
        device = resolve_device(cfg)
        key = (str(cfg.run.dataset), int(cfg.data.pool_seed), int(cfg.data.global_val_size),
               float(cfg.data.client_train_frac), str(device))
        if key not in self._data:
            bundle = load_dataset(str(cfg.run.dataset), str(cfg.run.data_dir), device)
            self._data[key] = (bundle, build_pools(bundle, cfg), device)
        return self._data[key]

    def env(self, cfg: Config, key: Any) -> FLEnv:
        if key not in self._envs:
            bundle, pools, device = self.data(cfg)
            self._envs[key] = FLEnv(cfg, bundle, pools, device)
        return self._envs[key]

def _find_tuner(name: str):
    """Locate tune_settings / tune_alpha (Part 7 does not state the module; search its controller modules)."""
    for mod in ("fixed", "heuristics", "extended", "bandit", "base"):
        m = importlib.import_module(f"fedrl.controllers.{mod}")
        if hasattr(m, name):
            return getattr(m, name)
    raise ImportError(f"{name} not found in fedrl.controllers.*; import it explicitly in fedrl/runner.py")


def _ensure_tuning(job: Job, cfg_a: Config, rd: Path) -> None:
    """tuned_settings / tuned_alpha need {run_dir}/tuning.json before the controller is built."""
    if job.method not in ("tuned_settings", "tuned_alpha") or (rd / "tuning.json").exists():
        return
    _find_tuner("tune_settings" if job.method == "tuned_settings" else "tune_alpha")(cfg_a, run_dir=rd)


def build_run_controller(job: Job, base_cfg: Config, for_eval: bool = False):
    """(controller, method+variant-applied cfg, run dir) for a job."""
    cfg_a = job_config(job, base_cfg, for_eval=for_eval, apply=True)
    rd = job_run_dir(job, base_cfg)
    _ensure_tuning(job, cfg_a, rd)
    return build_controller(job.method, cfg_a, state_dim(cfg_a), rd), cfg_a, rd


def _validate(job: Job, base_cfg: Config, ctx: Context) -> None:
    ctrl, cfg_a, rd = build_run_controller(job, base_cfg)
    bundle, pools, device = ctx.data(cfg_a)
    res = evaluate_on_validation(ctrl, cfg_a, bundle, pools, device, env=ctx.env(cfg_a, ("val", str(rd))))
    save_validation_result(rd, res, {"method": job.method, "dataset": job.dataset,
                                     "variant": job.variant or "default", "train_seed": job.train_seed})


def execute_job(job: Job, base_cfg: Config, ctx: Context, force: bool = False) -> str:
    """Run one job. Returns 'skipped' (already done) or 'done'."""
    if not force and job_status(job, base_cfg) == "done":
        return "skipped"
    if job.kind == "train":
        train(job_config(job, base_cfg), job.method, job.variant or "default", job.train_seed)
        if not is_rl(job.method):  # learning baseline (bandit): record its validation score
            _validate(job, base_cfg, ctx)
        return "done"
    if job.kind == "tune":
        _validate(job, base_cfg, ctx)
        return "done"
    if job.kind == "eval":
        ctrl, cfg_a, rd = build_run_controller(job, base_cfg, for_eval=True)
        if is_trained(job.method):
            ck = CheckpointManager(rd)
            if ck.load_best() is None and ck.load_latest() is None:
                raise RuntimeError(f"run not trained yet (no checkpoint): {rd}")
        bundle, pools, device = ctx.data(cfg_a)
        evaluate_and_save(ctrl, job.scenario, int(job.eval_seed), cfg_a, bundle, pools, device, rd,
                          _meta(job), tag=job.tag, force=force,
                          env=ctx.env(cfg_a, ("eval", str(rd), job.eval_overrides)))
        return "done"
    raise ValueError(f"unknown job kind '{job.kind}'")

def _meta(job: Job) -> dict:
    return {"experiment": job.experiment, "method": job.method, "variant": job.variant or "default",
            "train_seed": job.train_seed}


def run_jobs(jobs: list[Job], base_cfg: Config, force: bool = False, continue_on_error: bool = True) -> dict:
    """Run jobs sequentially (train/tune first, then eval, as given); finished jobs are skipped."""
    ctx = Context()
    order = {"train": 0, "tune": 1, "eval": 2}
    jobs = sorted(jobs, key=lambda j: order[j.kind])
    stats = {"done": 0, "skipped": 0, "failed": []}
    for job in tqdm(jobs, desc="jobs", unit="job"):
        try:
            stats[execute_job(job, base_cfg, ctx, force)] += 1
        except Exception as exc:  # noqa: BLE001 - report and continue with other jobs
            logger.exception("job failed: %s", format_job(job))
            stats["failed"].append((format_job(job), repr(exc)))
            if not continue_on_error:
                raise
    return stats


def spawn_workers(script: str, args: list[str], n: int) -> list[int]:
    """Start n subprocesses (one thread each), worker w gets FEDRL_WORKER='w/n'; returns exit codes."""
    procs = []
    for w in range(n):
        env = dict(os.environ, FEDRL_WORKER=f"{w}/{n}", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1")
        procs.append(subprocess.Popen([sys.executable, script, *args], env=env))
    return [p.wait() for p in procs]
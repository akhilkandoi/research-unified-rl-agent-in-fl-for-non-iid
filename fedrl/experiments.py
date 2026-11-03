"""Experiment definitions (pilot, E1-E7), job lists, sharding and job status (MASTER 2.14, 2.15)."""
from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from fedrl.config import Config, load_config
from fedrl.data.scenario_sets import ScenarioSpec, test_scenarios
from fedrl.eval.metrics import result_path, validation_path
from fedrl.methods import apply_method, apply_variant, is_rl
from fedrl.runs.checkpoint import CheckpointManager
from fedrl.runs.run_manager import run_dir as _run_dir

RL_MAIN = ["joint", "stacked_shared", "stacked_specialized"]
NON_RL = ["fedavg", "fixed_alpha_0.5", "fixed_alpha_0.1", "fedprox", "loss_heuristic",
          "tuned_settings", "tuned_alpha", "bandit"]
ABLATIONS = ["joint_mem_none", "joint_no_select", "joint_no_train", "joint_no_mix", "joint_gru_window"]
E2B_METHODS = ["single_select", "single_train", "single_mix", "joint_ema_beta0.5", "joint_ema_delta",
               "joint_separate", "joint_no_attention", "joint_extra_features"]
E7_METHODS = ["joint", "stacked_shared", "stacked_specialized", "fedavg", "fixed_alpha_0.5", "loss_heuristic"]
E3_SETTINGS = [(0.1, 0.3), (0.0, 0.3), (0.3, 0.3), (0.1, 0.0), (0.0, 0.0)]  # (w_comm, w_fair); first = default
PILOT_UPDATE_SWEEP = [2, 4, 8]
PILOT_SEEDS = [0, 1, 2]
PILOT_BASELINES = ["fixed_alpha_0.5", "loss_heuristic", "bandit"]
PILOT_CONFIG = "configs/pilot.yaml"
LEARNING_BASELINES = ["bandit"]  # non-RL, but trained by train() (LinUCB state)


def is_trained(method: str) -> bool:
    """True if train() trains this method (RL policies and learning baselines)."""
    return is_rl(method) or method in LEARNING_BASELINES
FOCUS_A = 0.2

EXPERIMENT_TIERS = {"pilot": "core", "E1": "core", "E2": "core", "E3": "core", "E4": "core",
                    "E5": "core", "E2b": "extended", "E6": "extended", "E7": "extended"}


@dataclass(frozen=True)
class Job:
    """One unit of work. kind: train (RL training), tune (baseline tuning + validation), eval (test)."""
    experiment: str
    dataset: str
    method: str
    variant: str
    train_seed: int
    scenario: ScenarioSpec | None
    eval_seed: int | None
    kind: str
    tier: str = "core"
    overrides: tuple[str, ...] = ()        # run-defining overrides (change the config hash)
    eval_overrides: tuple[str, ...] = ()   # evaluation-only overrides (e.g. N=50), not in run folder
    config_file: str | None = None
    tag: str = ""


# ----------------------------------------------------------------------------- keys / config
def run_key(job: Job) -> tuple:
    """Identity of the run folder this job trains or reads from."""
    return (job.dataset, job.method, job.variant, job.train_seed, job.overrides, job.config_file)


def job_key(job: Job) -> str:
    """Unique string key of a job (used for de-duplication across experiments)."""
    sc = "-" if job.scenario is None else f"{job.scenario.a:g}/{job.scenario.drift}/{job.scenario.seed}"
    return "|".join(map(str, [job.kind, *run_key(job), sc, job.eval_seed, job.eval_overrides, job.tag]))


def format_job(job: Job) -> str:
    """One-line description for dry runs."""
    sc = "" if job.scenario is None else f" a={job.scenario.a:g} drift={job.scenario.drift} seed={job.scenario.seed}"
    return f"[{job.experiment}] {job.kind:5s} {job.dataset}/{job.method}/{job.variant} train_seed={job.train_seed}{sc}"


def _derive_cfg(base_cfg: Config, config_file: str | None, overrides: Iterable[str]) -> Config:
    """Config for a job: base cfg (or job.config_file, inheriting run.*) plus overrides."""
    ov = list(overrides)
    if config_file:
        keep = [f"run.{k}={base_cfg.get('run.' + k)}" for k in ("runs_dir", "data_dir", "device")
                if base_cfg.get("run." + k, None) is not None]
        return load_config(config_file, keep + ov)
    fd, tmp = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(base_cfg.to_dict(), f)
        return load_config(tmp, ov)
    finally:
        os.unlink(tmp)


def job_config(job: Job, base_cfg: Config, for_eval: bool = False, apply: bool = False) -> Config:
    """Config of a job. for_eval adds eval_overrides; apply resolves method AND variant (Part 6)."""
    ov = list(job.overrides) + (list(job.eval_overrides) if for_eval else [])
    cfg = _derive_cfg(base_cfg, job.config_file, ov)
    if apply:
        cfg = apply_variant(apply_method(cfg, job.method), job.variant or "default")
    return cfg

def job_run_dir(job: Job, base_cfg: Config) -> Path:
    """Run folder of a job; Part 6 needs the RESOLVED (method+variant applied) training config."""
    return _run_dir(job_config(job, base_cfg, apply=True), job.method, job.variant or "default", job.train_seed)



# ----------------------------------------------------------------------------- builders
def _ds(dataset: str) -> tuple[str, ...]:
    return (f"run.dataset={dataset}",)


def _scenarios(cfg: Config, focus: bool) -> list[ScenarioSpec]:
    specs = test_scenarios(cfg)
    if not focus:
        return specs
    avals = sorted({s.a for s in specs})
    a0 = FOCUS_A if any(abs(a - FOCUS_A) < 1e-9 for a in avals) else avals[len(avals) // 2]
    return [s for s in specs if abs(s.a - a0) < 1e-9]

def _method_jobs(exp: str, dataset: str, methods: list[str], seeds: list[int], scenarios: list[ScenarioSpec],
                 variant: str = "default", extra_overrides: tuple[str, ...] = (), eval_overrides: tuple[str, ...] = (),
                 tag: str = "", with_train: bool = True, config_file: str | None = None) -> list[Job]:
    tier = EXPERIMENT_TIERS[exp]
    ov = _ds(dataset) + extra_overrides
    trains: list[Job] = []
    evals: list[Job] = []
    for m in methods:
        trained = is_trained(m)
        for s in (seeds if trained else [0]):
            if with_train:
                trains.append(Job(exp, dataset, m, variant, s, None, None, "train" if trained else "tune",
                                  tier, ov, (), config_file))
            for sp in scenarios:
                evals.append(Job(exp, dataset, m, variant, s, sp, sp.seed, "eval", tier, ov, eval_overrides,
                                 config_file, tag))
    return trains + evals

def _pilot(cfg: Config) -> list[Job]:
    jobs: list[Job] = []
    for u in PILOT_UPDATE_SWEEP:
        ov = _ds("mnist") + (f"ppo.checkpoint_every={math.lcm(10, u)}", f"ppo.validate_every={math.lcm(20, u)}")
        for s in PILOT_SEEDS:  # variant token upd<n> sets ppo.update_every_episodes (Part 6)
            jobs.append(Job("pilot", "mnist", "joint", f"upd{u}", s, None, None, "train", "core", ov, (), PILOT_CONFIG))
    for s in PILOT_SEEDS:
        jobs.append(Job("pilot", "mnist", "stacked_shared", "upd2", s, None, None, "train", "core",
                        _ds("mnist"), (), PILOT_CONFIG))
    for b in PILOT_BASELINES:
        kind = "train" if b in LEARNING_BASELINES else "tune"
        jobs.append(Job("pilot", "mnist", b, "default", 0, None, None, kind, "core", _ds("mnist"), (), PILOT_CONFIG))
    return jobs

def _e1(cfg: Config) -> list[Job]:
    return _method_jobs("E1", "mnist", RL_MAIN + NON_RL, [0, 1, 2], _scenarios(cfg, False))


def _e2(cfg: Config) -> list[Job]:
    return _method_jobs("E2", "mnist", ABLATIONS, [0, 1], _scenarios(cfg, True))


def _e3(cfg: Config) -> list[Job]:
    jobs: list[Job] = []
    for i, (wc, wf) in enumerate(E3_SETTINGS):
        variant = "default" if i == 0 else f"wcomm{wc:g}_wfair{wf:g}"  # Part 6 variant tokens
        jobs += _method_jobs("E3", "mnist", ["joint"], [0], _scenarios(cfg, True), variant)
    return jobs

def _e4(cfg: Config) -> list[Job]:
    return _method_jobs("E4", "fmnist", RL_MAIN + NON_RL, [0, 1, 2], _scenarios(cfg, True))


def _e5(cfg: Config) -> list[Job]:
    return []  # free: computed from logs of E1 only


def _e2b(cfg: Config) -> list[Job]:
    return _method_jobs("E2b", "mnist", E2B_METHODS, [0, 1], _scenarios(cfg, True))


def _e6(cfg: Config) -> list[Job]:
    return _method_jobs("E6", "mnist", RL_MAIN, [0, 1, 2], _scenarios(cfg, True), eval_overrides=("fl.N=50", "fl.k=10"),
                        tag="N50", with_train=False)


def _e7(cfg: Config) -> list[Job]:
    return _method_jobs("E7", "cifar10", E7_METHODS, [0, 1], _scenarios(cfg, True))


_BUILDERS = {"pilot": _pilot, "E1": _e1, "E2": _e2, "E3": _e3, "E4": _e4, "E5": _e5,
             "E2b": _e2b, "E6": _e6, "E7": _e7}


def jobs_for(experiment: str, cfg: Config) -> list[Job]:
    """All jobs of one experiment (train/tune jobs first, then eval jobs)."""
    if experiment not in _BUILDERS:
        raise ValueError(f"unknown experiment '{experiment}'; choose from {sorted(_BUILDERS)}")
    return _BUILDERS[experiment](cfg)


def experiments_in_tier(tier: str | None) -> list[str]:
    """Experiment names of a tier ('core', 'extended') or all if tier is None/'all'."""
    return [e for e, t in EXPERIMENT_TIERS.items() if tier in (None, "all", t)]


def dedupe(jobs: Iterable[Job]) -> list[Job]:
    """Remove duplicate jobs (e.g. E3 default = E1 joint seed 0), keeping first occurrence."""
    seen, out = set(), []
    for j in jobs:
        k = job_key(j)
        if k not in seen:
            seen.add(k)
            out.append(j)
    return out


def count_jobs(jobs: Iterable[Job]) -> dict[str, int]:
    """Counts per kind plus total."""
    c = {"train": 0, "tune": 0, "eval": 0}
    for j in jobs:
        c[j.kind] += 1
    c["total"] = sum(c.values())
    return c


# ----------------------------------------------------------------------------- sharding
def shard_jobs(jobs: list[Job], shard: tuple[int, int] | None) -> list[Job]:
    """Deterministic sharding: run r (in sorted order of run keys) belongs to shard r mod n.

    Every job inherits the shard of the run it trains or reads, so evaluation happens on the
    machine that trained the run and each job lands in exactly one shard.
    """
    if shard is None:
        return list(jobs)
    i, n = shard
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"invalid shard {i}/{n}")
    order = {k: r for r, k in enumerate(sorted({run_key(j) for j in jobs}, key=str))}
    return [j for j in jobs if order[run_key(j)] % n == i]


# ----------------------------------------------------------------------------- status
def _log_last_episode(path: Path) -> int:
    if not path.exists():
        return 0
    last = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                last = max(last, int(float(row["episode"])))
            except (KeyError, ValueError):
                continue
    return last


def _train_progress(rd: Path) -> int:
    ep = _log_last_episode(rd / "train_log.csv")
    if rd.exists():
        try:
            st = CheckpointManager(rd).load_latest()
            if st is not None:
                ep = max(ep, int(st.get("episode", 0)))
        except Exception:  # corrupt checkpoint: rely on the log
            pass
    return ep


def job_status(job: Job, cfg: Config | None = None) -> str:
    """'done' | 'partial' | 'not_started' from checkpoints, logs and result files."""
    base = cfg if cfg is not None else load_config(None)
    rd = job_run_dir(job, base)
    if job.kind == "eval":
        return "done" if result_path(rd, job.scenario, job.eval_seed, job.tag).exists() else "not_started"
    if job.kind == "tune":
        if validation_path(rd).exists():
            return "done"
        return "partial" if rd.exists() and any(rd.iterdir()) else "not_started"
    if job.kind == "train":
        ep = _train_progress(rd)
        target = int(job_config(job, base).ppo.episodes)
        if ep < target:
            return "partial" if ep > 0 else "not_started"
        if not is_rl(job.method) and not validation_path(rd).exists():
            return "partial"  # learning baseline trained, validation score still missing
        return "done"
    raise ValueError(f"unknown job kind '{job.kind}'")
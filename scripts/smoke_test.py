"""End-to-end smoke test: train, kill+resume, baselines, evaluate, analysis. Exit code 1 on any failure."""
# ===== EDIT HERE =====
CONFIG = "configs/smoke.yaml"        # tiny config
USE_SYNTHETIC_DATA = True            # True: offline synthetic images; False: real dataset from run.data_dir
DATASET = "mnist"                    # mnist | fmnist | cifar10
SEED = 0                             # training seed
RL_METHOD = "joint"                  # RL method trained and evaluated
TRAIN_EPISODES = 2                   # episodes of the uninterrupted run
KILL_AFTER_EPISODE = 1               # the interrupted run stops here, then resumes to TRAIN_EPISODES
BASELINE_EPISODES = 1                # episodes per non-RL baseline
EVAL_A = 0.2                         # test-scenario skew (unseen value)
EVAL_DRIFT = "abrupt"                # test-scenario drift
EVAL_SEEDS = [1000, 1001]            # evaluation seeds
OVERRIDES = []                       # extra config overrides, e.g. ["fl.k=3"]
RUN_ANALYZE_SCRIPT = False           # also run scripts/analyze.py (needs its own results layout)
KEEP_TMP = False                     # keep the temporary runs directory
STOP_ON_FAIL = False                 # stop at first failing stage
# ===== END EDIT =====

import contextlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fedrl.config import load_config, validate  # noqa: E402
from fedrl.controllers import build_controller  # noqa: E402
import os
import fedrl.data.datasets as _datasets  # noqa: E402  (module attribute, so the synthetic patch applies)
from fedrl.controllers.base import rollout_episode  # noqa: E402
from fedrl.utils.seeding import make_rng  # noqa: E402
from fedrl.data.pools import build_pools  # noqa: E402
from fedrl.data.scenario_sets import ScenarioSpec  # noqa: E402
from fedrl.env.fl_env import FLEnv  # noqa: E402
from fedrl.env.state import state_dim  # noqa: E402
from fedrl.eval.evaluate_policy import evaluate_controller  # noqa: E402
from fedrl.eval.stats import paired_bootstrap  # noqa: E402
from fedrl.methods import all_method_names, apply_method, is_rl  # noqa: E402
from fedrl.rl.trainer import train  # noqa: E402
from fedrl.runs.checkpoint import CheckpointManager  # noqa: E402
from fedrl.utils.testing import (block_official_test, compare_checkpoints, compare_run_logs,  # noqa: E402
                                 install_synthetic_data)

CTX: dict = {}


def _jd(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(type(obj))


def cfg_with(extra):
    tmp = CTX["tmp"]
    base = ["run.device=cpu", f"run.dataset={DATASET}", f"run.data_dir={tmp / 'data'}"]
    return load_config(str(ROOT / CONFIG), base + list(OVERRIDES) + list(extra))


def stage_config():
    cfg = cfg_with([f"run.runs_dir={CTX['tmp'] / 'runs_full'}", f"ppo.episodes={TRAIN_EPISODES}"])
    validate(cfg)
    CTX["cfg"], CTX["device"] = cfg, torch.device("cpu")
    return f"N={cfg.fl.N} k={cfg.fl.k} T={cfg.fl.T}"


def stage_data():
    cfg = CTX["cfg"]
    CTX["bundle"] = _datasets.load_dataset(cfg.run.dataset, cfg.run.data_dir, CTX["device"])
    CTX["pools"] = build_pools(CTX["bundle"], cfg)
    return f"train_x={tuple(CTX['bundle'].train_x.shape)}"

def stage_train():
    with block_official_test():  # any test-set read during training is fatal
        CTX["run_dir"] = Path(train(CTX["cfg"], RL_METHOD, "default", SEED))
    if CheckpointManager(CTX["run_dir"]).load_latest() is None:
        raise RuntimeError("no checkpoint written")
    return str(CTX["run_dir"].relative_to(CTX["tmp"]))


def stage_kill_resume():
    runs = CTX["tmp"] / "runs_resume"
    with block_official_test():
        cfg1 = cfg_with([f"run.runs_dir={runs}", f"ppo.episodes={KILL_AFTER_EPISODE}"])
        train(cfg1, RL_METHOD, "default", SEED)  # "killed" after the episode-1 checkpoint
        cfg2 = cfg_with([f"run.runs_dir={runs}", f"ppo.episodes={TRAIN_EPISODES}"])
        resumed = Path(train(cfg2, RL_METHOD, "default", SEED))
    a = CheckpointManager(CTX["run_dir"]).load_latest()
    b = CheckpointManager(resumed).load_latest()
    n, problems = compare_checkpoints(a, b)
    problems += compare_run_logs(CTX["run_dir"], resumed)
    if n == 0 or problems:
        raise RuntimeError(f"compared={n} problems={problems}")
    return f"{n} policy tensors identical, logs identical"

def stage_baselines():
    names = [m for m in all_method_names() if not is_rl(m)]
    notes = []
    data = (CTX["bundle"].train_x, CTX["bundle"].train_y)
    for name in names:
        cfg = apply_method(CTX["cfg"], name)
        env = FLEnv(cfg, CTX["bundle"], CTX["pools"], CTX["device"])
        try:
            ctrl = build_controller(name, cfg, env.state_dim, None)
        except Exception:  # noqa: BLE001
            if name.startswith("tuned_"):
                notes.append(f"{name}:needs-tuning-file")
                continue
            raise
        for ep in range(BASELINE_EPISODES):
            out = rollout_episode(env, ctrl, ScenarioSpec(0.3, "abrupt", ep),
                                  make_rng(ep, "eval_select"), True, False, data)
            if not np.isfinite(out["return"]):
                raise RuntimeError(f"{name} produced a non-finite return")
    return f"{len(names) - len(notes)}/{len(names)} baselines ran" + (f" ({', '.join(notes)})" if notes else "")

def stage_evaluate():
    results = {}
    out_dir = CTX["tmp"] / "results"
    for name in ("fedavg", "fixed_alpha_0.5", RL_METHOD):
        cfg_m = apply_method(CTX["cfg"], name)
        ctrl = build_controller(name, cfg_m, state_dim(cfg_m), run_dir=CTX["run_dir"] if is_rl(name) else None)
        for es in EVAL_SEEDS:
            spec = ScenarioSpec(EVAL_A, EVAL_DRIFT, es)
            res = evaluate_controller(ctrl, spec, es, cfg_m, CTX["bundle"], CTX["pools"], CTX["device"])
            path = out_dir / name / f"seed{es}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(res, default=_jd))
            results[(name, es)] = res
    CTX["results"] = results
    return f"{len(results)} evaluations written"


def _metric(res):
    for key, val in res.items():
        if "global" in key.lower() and "acc" in key.lower() and isinstance(val, (int, float)):
            return key
    for key, val in res.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return key
    raise RuntimeError("no scalar metric in evaluation result")


def stage_analysis():
    res = CTX["results"]
    key = _metric(res[(RL_METHOD, EVAL_SEEDS[0])])
    a = np.array([res[(RL_METHOD, s)][key] for s in EVAL_SEEDS], float)
    b = np.array([res[("fedavg", s)][key] for s in EVAL_SEEDS], float)
    out = paired_bootstrap(a, b, n_resamples=200, ci=0.95, rng=np.random.default_rng(0))
    missing = [k for k in ("mean_diff", "lo", "hi", "excludes_zero") if k not in out]
    if missing:
        raise RuntimeError(f"paired_bootstrap result lacks {missing}")
    importlib.import_module("fedrl.eval.analysis")
    if RUN_ANALYZE_SCRIPT:
        code = ("import runpy,sys;sys.path.insert(0,%r);"
                "from fedrl.utils.testing import install_synthetic_data;"
                "install_synthetic_data().__enter__();runpy.run_path(%r, run_name='__main__')"
                % (str(ROOT), str(ROOT / "scripts" / "analyze.py")))
        env = {**os.environ, "FEDRL_RUNS_DIR": str(CTX["tmp"] / "runs_full")}
        proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"analyze.py failed:\n{proc.stderr[-800:]}")
    return f"metric={key} mean_diff(joint-fedavg)={out['mean_diff']:.4f} (2 seeds, smoke only)"


STAGES = [("config", stage_config), ("data", stage_data), ("train", stage_train),
          ("kill+resume", stage_kill_resume), ("baselines", stage_baselines),
          ("evaluate", stage_evaluate), ("analysis", stage_analysis)]


def main() -> int:
    os.environ.pop("FEDRL_RUNS_DIR", None)
    CTX["tmp"] = Path(tempfile.mkdtemp(prefix="fedrl_smoke_"))
    failed = False
    guard = install_synthetic_data() if USE_SYNTHETIC_DATA else contextlib.nullcontext()
    try:
        with guard:
            for name, fn in STAGES:
                t0 = time.perf_counter()
                try:
                    detail, ok = fn() or "", True
                except Exception:  # noqa: BLE001
                    detail, ok = traceback.format_exc(limit=8), False
                print(f"[{'PASS' if ok else 'FAIL'}] {name:<12s} {time.perf_counter() - t0:7.1f}s  {detail}")
                failed |= not ok
                if not ok and STOP_ON_FAIL:
                    break
    finally:
        if KEEP_TMP:
            print(f"kept {CTX['tmp']}")
        else:
            shutil.rmtree(CTX["tmp"], ignore_errors=True)
    print("SMOKE TEST FAILED" if failed else "SMOKE TEST PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
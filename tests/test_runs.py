import os
import time

import pandas as pd
import pytest
import torch

from fedrl.config import load_config
from fedrl.runs.checkpoint import CheckpointManager
from fedrl.runs.locks import LockHeldError, RunLock, is_stale, shard_jobs
from fedrl.runs.run_manager import ConfigMismatchError, check_config_match, prepare_run_dir, run_dir
from fedrl.runs.train_log import TrainLog


def test_checkpoint_keep_two_best_and_fallback(tmp_path):
    m = CheckpointManager(tmp_path)
    for ep in (1, 2, 3):
        m.save({"x": torch.tensor([float(ep)])}, ep, is_best=(ep == 2))
    assert [e for e, _ in m.list_checkpoints()] == [2, 3]
    assert m.load_best()["episode"] == 2
    assert m.load_latest()["episode"] == 3
    m.path_for(3).write_bytes(b"garbage")
    assert m.load_latest()["episode"] == 2


def test_config_hash_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDRL_RUNS_DIR", str(tmp_path))
    c1 = load_config()
    rd = prepare_run_dir(c1, "joint", "default", 0)
    assert rd == run_dir(c1, "joint", "default", 0)
    prepare_run_dir(load_config(overrides=["ppo.episodes=999"]), "joint", "default", 0)  # allowed
    c2 = load_config(overrides=["fl.k=3"])
    with pytest.raises(ConfigMismatchError):
        check_config_match(rd, c2)


def test_runs_dir_resolution(tmp_path, monkeypatch):
    from fedrl.runs.run_manager import resolve_runs_dir
    monkeypatch.setenv("FEDRL_RUNS_DIR", str(tmp_path))
    assert resolve_runs_dir(load_config()) == tmp_path
    monkeypatch.delenv("FEDRL_RUNS_DIR")
    assert str(resolve_runs_dir(load_config())) in ("runs", "/content/drive/MyDrive/fedrl_runs")


def test_log_truncation(tmp_path):
    lg = TrainLog(tmp_path, ["joint"], 3)
    for e in range(1, 6):
        lg.log_episode({"episode": e, "return": float(e)})
        lg.log_selection(e, [1, 0, 1])
    lg.truncate_after(3)
    assert list(pd.read_csv(tmp_path / "train_log.csv")["episode"]) == [1, 2, 3]
    assert list(pd.read_csv(tmp_path / "selection_counts.csv")["episode"]) == [1, 2, 3]


def test_stale_lock_detection(tmp_path):
    p = tmp_path / "x.lock"
    a = RunLock(p, heartbeat_s=0.05)
    a.acquire()
    with pytest.raises(LockHeldError):
        RunLock(p).acquire()
    time.sleep(0.2)
    assert not is_stale(p)                         # heartbeat keeps it fresh
    a._stop.set(); a._thread.join()
    old = time.time() - 1000
    os.utime(p, (old, old))
    assert is_stale(p)
    b = RunLock(p)
    b.acquire()                                    # steals the stale lock
    b.release()
    assert not p.exists()


def test_sharding_deterministic_disjoint_covering():
    jobs = list(range(23))
    shards = [shard_jobs(jobs, i, 4) for i in range(4)]
    assert shards == [shard_jobs(jobs, i, 4) for i in range(4)]
    flat = [j for s in shards for j in s]
    assert sorted(flat) == jobs and len(flat) == len(set(flat))
    assert shards[1] == [1, 5, 9, 13, 17, 21]
    with pytest.raises(ValueError):
        shard_jobs(jobs, 4, 4)
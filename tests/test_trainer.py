import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fedrl.config import load_config
from fedrl.rl.trainer import TestAccessError, check_schedule, forbid_test_access, train
from fedrl.runs.checkpoint import CheckpointManager

ROOT = Path(__file__).resolve().parents[1]


def _cfg(extra=()):
    return load_config(str(ROOT / "configs/smoke.yaml"), ["run.device=cpu", *extra])


@pytest.fixture(autouse=True)
def _runs(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDRL_RUNS_DIR", str(tmp_path))


def test_resume_equals_straight():
    a = train(_cfg(["ppo.episodes=6"]), "joint", "straight", 0)
    train(_cfg(["ppo.episodes=3"]), "joint", "split", 0)
    b = train(_cfg(["ppo.episodes=6"]), "joint", "split", 0)
    sa, sb = CheckpointManager(a).load_latest(), CheckpointManager(b).load_latest()
    assert sa["episode"] == sb["episode"] == 6
    for k, v in sa["policy"].items():
        assert torch.allclose(v, sb["policy"][k], atol=1e-6), k
    ra = pd.read_csv(a / "train_log.csv")["return"].values
    rb = pd.read_csv(b / "train_log.csv")["return"].values
    assert len(ra) == len(rb) == 6
    assert np.allclose(ra, rb, atol=1e-6)


def test_checkpoints_only_at_update_boundaries_and_best_by_validation():
    cfg = _cfg(["ppo.episodes=8", "ppo.update_every_episodes=2", "ppo.checkpoint_every=2",
                "ppo.validate_every=2"])
    rd = train(cfg, "joint", "bnd", 0)
    mgr = CheckpointManager(rd)
    assert all(e % 2 == 0 for e, _ in mgr.list_checkpoints())
    val = pd.read_csv(rd / "val_log.csv")
    best = mgr.load_best()
    assert best["best"]["val_return"] == pytest.approx(val["val_return"].max())
    assert best["episode"] == int(val.loc[val["val_return"].idxmax(), "episode"])
    with pytest.raises(ValueError):
        check_schedule(_cfg(["ppo.update_every_episodes=2", "ppo.checkpoint_every=3"]))


def test_stacked_and_joint_same_episodes_and_steps():
    j = train(_cfg(["ppo.episodes=2"]), "joint", "default", 0)
    s = train(_cfg(["ppo.episodes=2"]), "stacked_shared", "default", 0)
    dj, ds = pd.read_csv(j / "train_log.csv"), pd.read_csv(s / "train_log.csv")
    cols = ["scenario_a", "drift", "scenario_seed", "env_steps"]
    assert dj[cols].equals(ds[cols])
    assert dj["env_steps"].iloc[-1] == 2 * _cfg().fl.T


def test_runtime_guard_blocks_test_access():
    import fedrl.data.scenario_sets as ss
    cfg = _cfg()
    with forbid_test_access():
        with pytest.raises(TestAccessError):
            ss.test_scenarios(cfg)
    assert len(ss.test_scenarios(cfg)) > 0  # restored afterwards


def test_static_no_test_access_in_training_code():
    files = [*(ROOT / "fedrl/rl").glob("*.py"), *(ROOT / "fedrl/runs").glob("*.py"),
             ROOT / "fedrl/methods.py", ROOT / "scripts/train.py"]
    banned = {"get_official_test", "test_scenarios"}
    for f in files:
        for node in ast.walk(ast.parse(f.read_text())):
            names = set()
            if isinstance(node, ast.ImportFrom):
                names = {a.name for a in node.names}
            elif isinstance(node, ast.Import):
                names = {a.name.split(".")[-1] for a in node.names}
            elif isinstance(node, ast.Name):
                names = {node.id}
            elif isinstance(node, ast.Attribute):
                names = {node.attr}
            assert not (names & banned), f"{f} touches {names & banned}"
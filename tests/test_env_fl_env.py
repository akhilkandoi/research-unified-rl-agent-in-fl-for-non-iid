"""Integration tests for FLEnv on a small synthetic dataset (no downloads)."""
import ast
from pathlib import Path

import pytest
import torch

from fedrl.config import load_config
from fedrl.data.datasets import DatasetBundle
from fedrl.data.pools import build_pools
from fedrl.data.scenario_sets import ScenarioSpec
from fedrl.env.fl_env import FLEnv
from fedrl.env.state import state_dim
from fedrl.fl.resources import drain

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")
INFO_KEYS = ["r_global", "r_pers", "r_fair", "comm_norm", "comp_norm", "final_bonus", "comm", "comp",
             "latency", "mean_alpha", "selected", "pers_gain_chosen", "pers_gain_all", "global_val_acc",
             "rounds_since_local", "n_probes", "return_so_far"]


def make_cfg(*overrides):
    return load_config(str(ROOT / "configs" / "smoke.yaml"), ["run.device=cpu", *overrides])


def synthetic_bundle(n_train=6000, n_test=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(10, 1, 28, 28, generator=g)
    def make(n):
        y = torch.randperm(n, generator=g) % 10
        x = protos[y] + 0.5 * torch.randn(n, 1, 28, 28, generator=g)
        return x.float(), y.long()
    tx, ty = make(n_train)
    ex, ey = make(n_test)
    return DatasetBundle(name="mnist", num_classes=10, train_x=tx, train_y=ty, test_x=ex, test_y=ey,
                         device=DEVICE)


@pytest.fixture(scope="module")
def bundle():
    return synthetic_bundle()


@pytest.fixture(scope="module")
def pools(bundle):
    return build_pools(bundle, make_cfg())


def make_env(bundle, pools, *overrides):
    cfg = make_cfg(*overrides)
    return cfg, FLEnv(cfg, bundle, pools, DEVICE)


def scripted_action(obs, k):
    idx = obs["eligible"].nonzero().flatten()[:k]
    return {"selected": idx.clone()}


def rollout(env, spec):
    obs = env.reset(spec)
    states, rewards, infos, done = [obs["state"].clone()], [], [], False
    while not done:
        obs, r, done, info = env.step(scripted_action(obs, env.k))
        states.append(obs["state"].clone()); rewards.append(r); infos.append(info)
    return states, rewards, infos


def test_reset_step_shapes_and_done(bundle, pools):
    cfg, env = make_env(bundle, pools)
    obs = env.reset(ScenarioSpec(a=0.3, drift="none", seed=1))
    N, D = env.N, state_dim(cfg)
    assert env.state_dim == D and obs["state"].shape == (N, D)
    assert obs["eligible"].shape == (N,) and obs["eligible"].dtype == torch.bool
    assert obs["epochs_mask"].shape == (N, 3) and obs["round"] == 0 and "window" not in obs
    for t in range(1, env.T + 1):
        obs, r, done, info = env.step(scripted_action(obs, env.k))
        assert obs["state"].shape == (N, D) and isinstance(r, float)
        assert done == (t == env.T) and obs["round"] == t
        for key in INFO_KEYS:
            assert key in info, key
        assert info["rounds_since_local"].shape == (N,)
    with pytest.raises(RuntimeError):
        env.step({})


@pytest.mark.parametrize("drift", ["none", "abrupt", "gradual"])
def test_n_probes(bundle, pools, drift):
    _, env = make_env(bundle, pools)
    _, _, infos = rollout(env, ScenarioSpec(a=0.3, drift=drift, seed=2))
    assert infos[-1]["n_probes"] == 1 + env.T + env.drift_events
    if drift == "none":
        assert env.drift_events == 0
    elif drift == "abrupt":
        assert env.drift_events == 1
    else:
        assert env.drift_events >= 1


def test_identical_seeds_identical_rollouts(bundle, pools):
    spec = ScenarioSpec(a=0.3, drift="abrupt", seed=3)
    _, env1 = make_env(bundle, pools)
    _, env2 = make_env(bundle, pools)
    s1, r1, _ = rollout(env1, spec)
    s2, r2, _ = rollout(env2, spec)
    assert r1 == pytest.approx(r2, abs=1e-6)
    for a, b in zip(s1, s2):
        assert torch.allclose(a, b, atol=1e-6)


def test_masks_under_low_battery(bundle, pools):
    cfg, env = make_env(bundle, pools)
    env.reset(ScenarioSpec(a=0.3, drift="none", seed=4))
    w = env.world
    mb = float(cfg.clients.min_battery)
    w.battery[0] = mb                       # drain(1) > 0 -> ineligible
    n2 = float(w.n[2])
    d1, d5 = drain(1, n2, cfg), drain(5, n2, cfg)
    w.battery[2] = mb + d1 + 0.5 * (d5 - d1)  # epochs 1 ok, 5 masked
    w.battery[3] = 1.0
    obs = env._make_obs()
    assert not bool(obs["eligible"][0]) and bool(obs["eligible"][2]) and bool(obs["eligible"][3])
    assert bool(obs["epochs_mask"][2, 0]) and not bool(obs["epochs_mask"][2, 2])
    assert obs["epochs_mask"][3].all()


def test_fewer_than_k_eligible(bundle, pools):
    _, env = make_env(bundle, pools)
    env.reset(ScenarioSpec(a=0.3, drift="none", seed=5))
    env.world.battery[:] = 0.0
    env.world.battery[3] = 1.0
    _, r, _, info = env.step({"selected": torch.tensor([0, 1])})
    assert info["selected"] == [3] and info["selection_repaired"] and r == r


def test_action_repair_and_masked_epochs(bundle, pools):
    _, env = make_env(bundle, pools)
    obs = env.reset(ScenarioSpec(a=0.3, drift="none", seed=6))
    act = {"selected": torch.tensor([0, 0, 99, -1]), "epochs_idx": torch.full((env.N,), 2),
           "lr_idx": torch.full((env.N,), 7)}
    _, _, _, info = env.step(act)
    assert len(info["selected"]) == env.k and len(set(info["selected"])) == env.k


def test_alpha_zero_gives_zero_pers_in_env(bundle, pools):
    _, env = make_env(bundle, pools)
    obs = env.reset(ScenarioSpec(a=0.3, drift="none", seed=7))
    done = False
    while not done:
        act = scripted_action(obs, env.k)
        act["alpha_value"] = torch.zeros(env.N)
        obs, _, done, info = env.step(act)
        assert info["r_pers"] == 0.0 and info["pers_gain_chosen"] == 0.0


def test_gru_window_shape(bundle, pools):
    cfg, env = make_env(bundle, pools, "state.memory_mode=gru_window")
    obs = env.reset(ScenarioSpec(a=0.3, drift="none", seed=8))
    D = state_dim(cfg)
    assert obs["window"].shape == (env.N, 5, D)
    assert torch.equal(obs["window"][:, -1], obs["state"]) and obs["window"][:, 0].abs().max() == 0
    first = obs["state"].clone()
    obs, _, _, _ = env.step(scripted_action(obs, env.k))
    assert obs["window"].shape == (env.N, 5, D) and torch.equal(obs["window"][:, -2], first)


def test_extra_features_run(bundle, pools):
    cfg, env = make_env(bundle, pools, "state.extra_features=true", "state.memory_mode=ema_delta")
    obs = env.reset(ScenarioSpec(a=0.3, drift="none", seed=9))
    assert obs["state"].shape == (env.N, 31)
    obs, _, _, _ = env.step(scripted_action(obs, env.k))
    assert torch.isfinite(obs["state"]).all()


def test_env_modules_never_touch_official_test_set():
    forbidden = {"get_official_test", "test_x", "test_y", "personal_test_idx"}
    for rel in ["fedrl/env/fl_env.py", "fedrl/env/state.py", "fedrl/env/memory.py", "fedrl/env/reward.py"]:
        tree = ast.parse((ROOT / rel).read_text())
        seen = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                seen.add(node.id)
            elif isinstance(node, ast.Attribute):
                seen.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for al in node.names:
                    seen.add(al.name.split(".")[-1])
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("fedrl.eval"):
                    seen.add("fedrl.eval")
        assert not (seen & forbidden), (rel, seen & forbidden)
        assert "fedrl.eval" not in seen, rel
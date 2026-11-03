"""Validity, determinism and registry tests for all controllers."""
import json

import numpy as np
import pytest
import torch

from fedrl.config import load_config
from fedrl.controllers import build_controller
from fedrl.controllers.base import (COMPUTE_SPEED_IDX, PROBE_LOSS_IDX, make_action,
                                    synthetic_obs)
from fedrl.env.state import state_dim
from fedrl.methods import all_method_names, is_rl

BASELINES = ["fedavg", "fixed_alpha_0.5", "fixed_alpha_0.1", "fedprox", "loss_heuristic",
             "tuned_settings", "tuned_alpha", "bandit", "value_selector", "adaptive_mixing"]


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/smoke.yaml")


def _tuning_dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    chosen = ({"lr_idx": 2, "epochs_idx": 0, "batch_idx": 1} if name == "tuned_settings"
              else {"alpha_idx": 3})
    (d / "tuning.json").write_text(json.dumps({"method": name, "chosen": chosen}))
    return d


def _build(name, cfg, tmp_path):
    return build_controller(name, cfg, state_dim(cfg), run_dir=_tuning_dir(tmp_path, name)
                            if name.startswith("tuned_") else None)


def check_action(a, obs, cfg):
    n, k = int(cfg.fl.N), int(cfg.fl.k)
    elig = obs["eligible"].cpu().numpy()
    sel = a["selected"].cpu().tolist()
    assert len(sel) == len(set(sel)) == min(k, int(elig.sum()))
    assert all(elig[i] for i in sel)
    for key in ("lr_idx", "epochs_idx", "batch_idx", "alpha_idx"):
        assert a[key].shape == (n,) and a[key].dtype == torch.long
    em = obs["epochs_mask"].cpu().numpy()
    for i in range(n):
        if em[i].any():
            assert em[i, int(a["epochs_idx"][i])]


@pytest.mark.parametrize("name", BASELINES)
@pytest.mark.parametrize("n_inel", [0, 2, 5])
def test_valid_actions(name, n_inel, cfg, tmp_path):
    rng = np.random.default_rng(1)
    ctrl = _build(name, cfg, tmp_path)
    obs = synthetic_obs(cfg, state_dim(cfg), rng, n_ineligible=n_inel, mask_top_epochs=True)
    ctrl.reset(obs)
    for det in (True, False):
        check_action(ctrl.act(obs, rng, deterministic=det), obs, cfg)


def test_fedavg_fedprox_and_fixed_alpha(cfg, tmp_path):
    rng = np.random.default_rng(0)
    obs = synthetic_obs(cfg, state_dim(cfg), rng)
    a = build_controller("fedavg", cfg, state_dim(cfg)).act(obs, rng)
    assert torch.all(a["alpha_value"] == 1.0) and "prox_mu" not in a
    p = build_controller("fedprox", cfg, state_dim(cfg)).act(obs, rng)
    assert p["prox_mu"] == pytest.approx(0.01) and torch.all(p["alpha_value"] == 1.0)
    for name, val in (("fixed_alpha_0.5", 0.5), ("fixed_alpha_0.1", 0.1)):
        act = build_controller(name, cfg, state_dim(cfg)).act(obs, rng)
        assert torch.allclose(act["alpha_value"], torch.full((cfg.fl.N,), val))
        assert cfg.menus.alpha[int(act["alpha_idx"][0])] == pytest.approx(val)


def test_tuned_controllers_use_tuning_json(cfg, tmp_path):
    rng = np.random.default_rng(0)
    obs = synthetic_obs(cfg, state_dim(cfg), rng)
    a = _build("tuned_settings", cfg, tmp_path).act(obs, rng)
    assert int(a["lr_idx"][0]) == 2 and int(a["batch_idx"][0]) == 1 and int(a["epochs_idx"][0]) == 0
    b = _build("tuned_alpha", cfg, tmp_path).act(obs, rng)
    assert int(b["alpha_idx"][0]) == 3


def test_tuned_without_tuning_json_raises(cfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        build_controller("tuned_alpha", cfg, state_dim(cfg), run_dir=tmp_path / "missing")


def test_masked_epochs_fall_back_to_largest_feasible(cfg):
    rng = np.random.default_rng(0)
    obs = synthetic_obs(cfg, state_dim(cfg), rng)
    n = int(cfg.fl.N)
    cases = [([True, True, False], 1), ([True, False, False], 0),
             ([False, True, True], 1), ([False, False, True], 2)]
    for row, expected in cases:
        obs["epochs_mask"] = torch.tensor([row] * n)
        a = make_action(torch.tensor([0, 1]), None, None, None, None, obs, cfg)
        assert int(a["epochs_idx"][0]) == expected
    obs["epochs_mask"] = torch.tensor([[True, False, True]] * n)  # default 1 masked -> largest
    a = make_action(torch.tensor([0]), None, None, None, None, obs, cfg)
    assert int(a["epochs_idx"][0]) == 2


def test_make_action_filters_ineligible_and_duplicates(cfg):
    rng = np.random.default_rng(0)
    obs = synthetic_obs(cfg, state_dim(cfg), rng, n_ineligible=2)
    bad = int(np.flatnonzero(~obs["eligible"].cpu().numpy())[0])
    a = make_action(torch.tensor([bad, 0, 0, 1, 2, 3, 4]), None, None, None, None, obs, cfg)
    assert bad not in a["selected"].tolist()
    assert len(set(a["selected"].tolist())) == len(a["selected"]) <= cfg.fl.k
    with pytest.raises(TypeError):
        make_action(torch.tensor([0]), None, None, None, None, obs, cfg, bogus=1)


def test_loss_heuristic_picks_top_losses_and_breaks_ties_by_speed(cfg):
    n, d = int(cfg.fl.N), state_dim(cfg)
    rng = np.random.default_rng(0)
    obs = synthetic_obs(cfg, d, rng)
    obs["state"][:, PROBE_LOSS_IDX] = torch.tensor([0.1, 0.9, 0.5, 0.7, 0.3, 0.8])
    ctrl = build_controller("loss_heuristic", cfg, d)
    assert ctrl.act(obs, rng)["selected"].tolist() == [1, 5]
    obs["eligible"][1] = False
    assert ctrl.act(obs, rng)["selected"].tolist() == [5, 3]
    obs["state"][:, PROBE_LOSS_IDX] = 0.5
    obs["state"][:, COMPUTE_SPEED_IDX] = torch.tensor([0.0, 0.0, 0.2, 0.9, 0.5, 0.1])
    obs["eligible"][:] = True
    assert ctrl.act(obs, rng)["selected"].tolist() == [3, 4]


@pytest.mark.parametrize("name", ["fedavg", "loss_heuristic", "bandit", "value_selector",
                                  "adaptive_mixing"])
def test_same_seed_same_actions(name, cfg, tmp_path):
    out = []
    for _ in range(2):
        rng = np.random.default_rng(7)
        ctrl = _build(name, cfg, tmp_path)
        obs = synthetic_obs(cfg, state_dim(cfg), rng, n_ineligible=1)
        ctrl.reset(obs)
        out.append(ctrl.act(obs, rng, deterministic=False))
    for key in out[0]:
        assert torch.equal(out[0][key], out[1][key])


def test_build_controller_covers_every_method(cfg, tmp_path):
    d = state_dim(cfg)
    for name in all_method_names():
        if is_rl(name):
            with pytest.raises(FileNotFoundError):  # dispatched to RLController, no checkpoint
                build_controller(name, cfg, d, run_dir=tmp_path / f"empty_{name}")
        else:
            assert build_controller(name, cfg, d, run_dir=_tuning_dir(tmp_path, name)
                                    if name.startswith("tuned_") else None).name
    with pytest.raises(KeyError):
        build_controller("no_such_method", cfg, d)


def test_rl_controller_loads_best_checkpoint_and_is_deterministic(cfg, tmp_path):
    from fedrl.policy.joint_policy import JointPolicy
    from fedrl.runs.checkpoint import CheckpointManager
    d = state_dim(cfg)
    torch.manual_seed(0)
    policy = JointPolicy(cfg, d)
    run = tmp_path / "joint_run"
    run.mkdir()
    CheckpointManager(run).save({"policy": policy.state_dict(), "config": cfg.to_dict()},
                                episode=1, is_best=True)
    ctrl = build_controller("joint", cfg, d, run_dir=run)
    rng = np.random.default_rng(3)
    obs = synthetic_obs(cfg, d, rng, n_ineligible=2, mask_top_epochs=True, device=ctrl.device)
    ctrl.reset(obs)
    a1 = ctrl.act(obs, rng)
    check_action(a1, obs, cfg)
    ctrl.reset(obs)
    a2 = ctrl.act(obs, rng)
    assert torch.equal(a1["selected"], a2["selected"])


@pytest.mark.parametrize("name", ["fedavg", "fixed_alpha_0.5", "fixed_alpha_0.1", "fedprox",
                                  "loss_heuristic", "tuned_settings", "tuned_alpha",
                                  "adaptive_mixing"])
def test_non_learning_controllers_have_no_state_dict(name, cfg, tmp_path):
    assert not hasattr(_build(name, cfg, tmp_path), "state_dict")  # Part 6 must not train them


@pytest.mark.parametrize("name", ["bandit", "value_selector"])
def test_learning_controllers_have_state_dict(name, cfg, tmp_path):
    ctrl = _build(name, cfg, tmp_path)
    assert hasattr(ctrl, "state_dict") and hasattr(ctrl, "load_state_dict")


def test_value_selector_marks_terminal_by_round(cfg):
    rng = np.random.default_rng(0)
    d, t = state_dim(cfg), int(cfg.fl.T)
    ctrl = build_controller("value_selector", cfg, d)
    obs = synthetic_obs(cfg, d, rng, round_t=t - 1)
    act = make_action(torch.tensor([0, 1]), None, None, None, None, obs, cfg)
    ctrl.observe(obs, act, 1.0, synthetic_obs(cfg, d, rng, round_t=t), {})       # no "done" key
    assert ctrl.buffer[-1][3] == 1.0
    ctrl.observe(obs, act, 1.0, synthetic_obs(cfg, d, rng, round_t=t - 1), {})
    assert ctrl.buffer[-1][3] == 0.0
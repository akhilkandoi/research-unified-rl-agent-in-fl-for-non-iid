"""Spec-coverage tests written against the INTERFACE CONTRACT (configs/smoke.yaml, CPU, synthetic data)."""
from __future__ import annotations

import ast
import inspect
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from fedrl.config import config_hash, load_config, validate
from fedrl.controllers import build_controller
from fedrl.data.pools import build_pools
from fedrl.data.scenario import Scenario
from fedrl.data.scenario_sets import ScenarioSpec
from fedrl.env.fl_env import FLEnv
from fedrl.env.memory import Memory
from fedrl.env.state import state_dim
from fedrl.eval.evaluate_policy import evaluate_controller
from fedrl.fl.client import local_train
from fedrl.fl.resources import drain, eligible_mask, epochs_mask
from fedrl.fl.server import aggregate
from fedrl.methods import all_method_names, apply_method, is_rl
from fedrl.models.cnn import build_model
from fedrl.policy.joint_policy import JointPolicy
from fedrl.policy.stacked import StackedPolicy
from fedrl.rl.buffer import RolloutBuffer
from fedrl.rl.ppo import PPOLearner
from fedrl.rl.trainer import train
from fedrl.runs.checkpoint import CheckpointManager
from fedrl.utils.flat import blend, get_flat, num_params, rel_div, set_flat, weighted_average
from fedrl.utils.seeding import make_torch_gen
from fedrl.utils.testing import (block_official_test, compare_checkpoints, compare_run_logs, default_action, install_synthetic_data, make_synthetic_bundle)
from fedrl.eval.metrics import to_jsonable
from fedrl.rl.trainer import check_schedule
from fedrl.utils.seeding import make_rng

ROOT = Path(__file__).resolve().parents[1]
SMOKE = str(ROOT / "configs" / "smoke.yaml")
DEV = torch.device("cpu")
SPEC = ScenarioSpec(a=0.3, drift="none", seed=0)
SPEC_DRIFT = ScenarioSpec(a=0.3, drift="abrupt", seed=0)
IDX_LOSS, IDX_MEM0, IDX_IDLE, IDX_PICKED, IDX_ROUND = 11, 19, 21, 22, 23
LOW_BATTERY = ["clients.battery_init_low=0.05", "clients.battery_init_high=0.07"]
INFO_KEYS = ["r_global", "r_pers", "r_fair", "comm_norm", "comp_norm", "final_bonus", "comm", "comp", "latency",
             "mean_alpha", "selected", "pers_gain_chosen", "pers_gain_all", "global_val_acc",
             "rounds_since_local", "n_probes", "return_so_far"]


def make_cfg(extra=(), runs_dir=None):
    overrides = ["run.device=cpu"] + list(extra)
    if runs_dir is not None:
        overrides.append(f"run.runs_dir={runs_dir}")
    return load_config(SMOKE, overrides)


def make_env(cfg, bundle, pools):
    return FLEnv(cfg, bundle, pools, DEV)


@pytest.fixture(autouse=True)
def _no_env_runs_dir(monkeypatch):
    monkeypatch.delenv("FEDRL_RUNS_DIR", raising=False)


@pytest.fixture(scope="session")
def bundle():
    return make_synthetic_bundle("mnist", "cpu")


@pytest.fixture(scope="session")
def pools(bundle):
    return build_pools(bundle, make_cfg())


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train 2 episodes once, with any read of the official test set made fatal."""
    runs = tmp_path_factory.mktemp("runs")
    cfg = make_cfg(["ppo.episodes=2"], runs)
    with install_synthetic_data(), block_official_test():
        run_dir = train(cfg, "joint", "default", 0)
    return cfg, Path(run_dir)


# ----------------------------------------------------------------------------- helpers
def rand_obs(cfg, seed=0, eligible=None, emask=None):
    g = torch.Generator().manual_seed(seed)
    n, d = cfg.fl.N, state_dim(cfg)
    return {"state": torch.randn(n, d, generator=g),
            "eligible": torch.ones(n, dtype=torch.bool) if eligible is None else eligible,
            "epochs_mask": torch.ones(n, 3, dtype=torch.bool) if emask is None else emask, "round": 0}


def batchify(obs):
    return {k: v.unsqueeze(0) for k, v in obs.items() if torch.is_tensor(v)}


def batch_action(action):
    keys = ("selected", "lr_idx", "epochs_idx", "batch_idx", "alpha_idx")
    return {k: action[k].unsqueeze(0) for k in keys}


def make_policy(cls, cfg):
    torch.manual_seed(0)
    return cls(cfg, state_dim(cfg)).eval()


# ----------------------------------------------------------------------------- data
def test_dirichlet_partition_and_size_bounds(bundle, pools):
    cfg = make_cfg(["fl.N=40"])
    sc = Scenario(ScenarioSpec(0.3, "none", 1), bundle, pools, cfg)
    assert sc.N == 40 and len(sc.clients) == 40
    train_pool = set(torch.cat(pools.train_by_class).tolist())
    val_pool = set(torch.cat(pools.val_by_class).tolist())
    gval = set(pools.global_val_idx.tolist())
    assert not (train_pool & val_pool) and not (train_pool & gval) and not (val_pool & gval)
    for c in sc.clients:
        assert cfg.clients.size_min <= c.n <= cfg.clients.size_max
        assert len(c.train_idx) == c.n and len(set(c.train_idx.tolist())) == c.n
        assert abs(len(c.val_idx) - max(round(cfg.clients.val_frac * c.n), cfg.clients.val_min)) <= 1
        assert set(c.train_idx.tolist()) <= train_pool and set(c.val_idx.tolist()) <= val_pool
        assert c.proportions.shape == (10,) and abs(float(c.proportions.sum()) - 1.0) < 1e-6
        assert (c.proportions >= 0).all()


def test_skew_parameter_controls_concentration(bundle, pools):
    cfg = make_cfg(["fl.N=40"])

    def stats(a):
        sc = Scenario(ScenarioSpec(a, "none", 3), bundle, pools, cfg)
        maxp = np.mean([c.proportions.max() for c in sc.clients])
        agree = np.mean([int(torch.bincount(bundle.train_y[c.train_idx], minlength=10).argmax()) ==
                         int(np.argmax(c.proportions)) for c in sc.clients])
        return maxp, agree

    strong, mild = stats(0.05), stats(1.0)
    assert strong[0] > mild[0] + 0.2
    assert strong[1] >= 0.8


def test_scenario_is_deterministic(bundle, pools):
    cfg = make_cfg()
    s1, s2 = (Scenario(SPEC, bundle, pools, cfg) for _ in range(2))
    for c1, c2 in zip(s1.clients, s2.clients):
        assert torch.equal(c1.train_idx, c2.train_idx) and torch.equal(c1.val_idx, c2.val_idx)
        assert np.allclose(c1.compute_speed, c2.compute_speed) and np.allclose(c1.battery0, c2.battery0)


def test_drift_flags_follow_spec(bundle, pools):
    cfg = make_cfg()
    t_total = cfg.fl.T

    def run(drift):
        sc = Scenario(ScenarioSpec(0.3, drift, 2), bundle, pools, cfg)
        before = np.array(sc.proportions_matrix(), dtype=float, copy=True)
        flags = [sc.apply_drift(t) for t in range(1, t_total + 1)]
        changed = not np.allclose(before, np.array(sc.proportions_matrix(), dtype=float))
        return flags, changed

    flags, changed = run("none")
    assert not any(flags) and not changed
    flags, changed = run("abrupt")
    assert flags == [t == cfg.drift.abrupt_round for t in range(1, t_total + 1)] and changed
    flags, changed = run("gradual")
    window = range(cfg.drift.gradual_start, cfg.drift.gradual_start + cfg.drift.gradual_rounds)
    assert flags == [t in window for t in range(1, t_total + 1)] and changed


# ----------------------------------------------------------------------------- math
def test_plackett_luce_probabilities_sum_to_one():
    cfg = make_cfg(["fl.N=3", "fl.k=2", "policy.active_heads.train=false", "policy.active_heads.mix=false"])
    pol = make_policy(JointPolicy, cfg)
    base = rand_obs(cfg)["state"]
    perms = torch.tensor(list(itertools.permutations(range(3), 2)))
    b = len(perms)
    ones = torch.ones(b, 3, dtype=torch.long)
    act_b = {"selected": perms, "lr_idx": ones, "epochs_idx": ones, "batch_idx": ones, "alpha_idx": 2 * ones}
    with torch.no_grad():
        full = {"state": base.unsqueeze(0).repeat(b, 1, 1), "eligible": torch.ones(b, 3, dtype=torch.bool),
                "epochs_mask": torch.ones(b, 3, 3, dtype=torch.bool)}
        p = torch.exp(pol.evaluate(full, act_b)[0]["joint"]).reshape(-1)
        assert torch.isfinite(p).all() and abs(float(p.sum()) - 1.0) < 1e-5
        part = dict(full, eligible=torch.tensor([True, True, False]).unsqueeze(0).repeat(b, 1))
        p2 = torch.exp(pol.evaluate(part, act_b)[0]["joint"]).reshape(-1)
    valid = ~(perms == 2).any(1)
    assert abs(float(p2[valid].sum()) - 1.0) < 1e-5
    assert (torch.nan_to_num(p2[~valid], nan=0.0) == 0).all()


def test_blend_arithmetic():
    g, l = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0])
    assert torch.allclose(blend(0.25, g, l), 0.25 * g + 0.75 * l)
    assert torch.equal(blend(0.0, g, l), l) and torch.equal(blend(1.0, g, l), g)
    out = blend(torch.tensor([0.0, 0.5, 1.0]), g, l)
    assert torch.allclose(out, torch.tensor([3.0, 2.0, 3.0]))
    assert abs(rel_div(1.1 * g, g) - 0.1) < 1e-5


def test_weighted_average_arithmetic():
    vecs = torch.tensor([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    expected = torch.full((3,), 4.0)
    assert torch.allclose(weighted_average(vecs, torch.tensor([100.0, 300.0])), expected)
    assert torch.allclose(weighted_average(vecs, torch.tensor([0.25, 0.75])), expected)

def _mom0(g):
    try:
        aggregate(torch.zeros(1, 4), torch.ones(1), g, torch.zeros_like(g), 0.0)
        return torch.zeros_like(g)
    except Exception:  # noqa: BLE001
        return None


def test_aggregate_arithmetic_with_server_momentum():
    g = torch.zeros(4)
    vecs = torch.tensor([[1.0] * 4, [3.0] * 4])
    n = torch.tensor([100.0, 300.0])
    g1, _ = aggregate(vecs, n, g, _mom0(g), 0.0)
    assert torch.allclose(g1, torch.full((4,), 2.5))
    g1, mom = aggregate(vecs, n, g, _mom0(g), 0.5)
    assert torch.allclose(g1, torch.full((4,), 2.5))
    g2, _ = aggregate(vecs, n, g1, mom, 0.5)  # delta = 0, v = 0.5 * 2.5
    assert torch.allclose(g2, torch.full((4,), 3.75))


def test_flat_roundtrip_and_param_count():
    model = build_model("mnist")
    assert num_params(model) == 46730
    vec = get_flat(model).clone()
    set_flat(model, vec * 2)
    assert torch.allclose(get_flat(model), vec * 2)


def test_local_train_changes_weights_and_is_reproducible(bundle):
    model = build_model("mnist")
    g = get_flat(model).clone()
    idx = torch.arange(64)

    def run(mu):
        return local_train(model, g, bundle.train_x, bundle.train_y, idx, 0.05, 1, 32, mu, make_torch_gen(0, "t"))

    a, b, prox = run(0.0), run(0.0), run(10.0)
    assert a.shape == g.shape and torch.equal(a, b) and not torch.allclose(a, g)
    assert float((prox - g).norm()) < float((a - g).norm())


def test_battery_drain_and_masks():
    cfg = make_cfg()
    assert abs(drain(2, 600, cfg) - 0.05) < 1e-9 and abs(drain(5, 100, cfg) - 5 * 100 * 0.05 / 1200) < 1e-9
    bat = torch.tensor([0.04, 0.08, 0.11, 0.9])
    n = torch.full((4,), 600.0)
    assert eligible_mask(bat, n, cfg).tolist() == [False, True, True, True]
    em = epochs_mask(bat, n, cfg)
    assert em[1].tolist() == [True, False, False] and em[2].tolist() == [True, True, False]
    assert em[3].tolist() == [True, True, True]


# ----------------------------------------------------------------------------- state / memory
def test_state_dim_modes():
    assert state_dim(make_cfg(["state.memory_mode=ema"])) == 24
    assert state_dim(make_cfg(["state.memory_mode=none"])) == 24
    assert state_dim(make_cfg(["state.memory_mode=ema_delta"])) == 26
    assert state_dim(make_cfg(["state.extra_features=true"])) > 24


def test_running_average_in_state(bundle, pools):
    cfg = make_cfg(["state.memory_mode=ema"])
    env = make_env(cfg, bundle, pools)
    obs0 = env.reset(SPEC)
    s0, beta, t_total = obs0["state"], cfg.state.ema_beta, cfg.fl.T
    assert torch.allclose(s0[:, IDX_MEM0], s0[:, IDX_LOSS], atol=1e-5)  # EMA starts at the first probe
    obs1, *_ = env.step(default_action(obs0, cfg))
    s1 = obs1["state"]
    assert torch.allclose(s1[:, IDX_MEM0], beta * s0[:, IDX_MEM0] + (1 - beta) * s1[:, IDX_LOSS], atol=1e-4)
    chosen = default_action(obs0, cfg)["selected"]
    others = torch.tensor([i for i in range(cfg.fl.N) if i not in chosen.tolist()], dtype=torch.long)
    assert torch.allclose(s1[chosen, IDX_PICKED], torch.full((len(chosen),), 1.0 / t_total), atol=1e-6)
    assert torch.allclose(s1[chosen, IDX_IDLE], torch.zeros(len(chosen)), atol=1e-6)
    assert torch.allclose(s1[others, IDX_IDLE], torch.full((len(others),), 1.0 / t_total), atol=1e-6)
    assert float(s1[0, IDX_ROUND]) > float(s0[0, IDX_ROUND]) >= 0.0 and torch.isfinite(s1).all()


def test_memory_none_zeroes_features(bundle, pools):
    cfg = make_cfg(["state.memory_mode=none"])
    env = make_env(cfg, bundle, pools)
    obs = env.reset(SPEC)
    assert (obs["state"][:, IDX_MEM0:IDX_MEM0 + 4] == 0).all()
    obs, *_ = env.step(default_action(obs, cfg))
    assert (obs["state"][:, IDX_MEM0:IDX_MEM0 + 4] == 0).all()
    mem = Memory(cfg, cfg.fl.N)
    mem.reset()
    assert mem.features().shape == (cfg.fl.N, 4)


# ----------------------------------------------------------------------------- policy
def test_policy_permutation_equivariance():
    cfg = make_cfg()
    pol = make_policy(JointPolicy, cfg)
    obs = rand_obs(cfg, seed=1)
    out = pol.sample(obs, make_torch_gen(0, "policy"))
    obs_b, act_b = batchify(obs), batch_action(out.action)
    perm = torch.tensor([3, 0, 5, 1, 2, 4])
    inv = torch.argsort(perm)
    obs_p = {k: v[:, perm] for k, v in obs_b.items()}
    act_p = {k: (inv[v] if k == "selected" else v[:, perm]) for k, v in act_b.items()}
    with torch.no_grad():
        lp, en, va = pol.evaluate(obs_b, act_b)
        lq, eq, vq = pol.evaluate(obs_p, act_p)
        for name in lp:
            assert torch.allclose(lp[name], lq[name], atol=1e-5)
            assert torch.allclose(en[name], eq[name], atol=1e-5)
            assert torch.allclose(va[name], vq[name], atol=1e-5)
        det = pol.sample(obs, make_torch_gen(0, "policy"), deterministic=True).action["selected"]
        det_p = pol.sample({**obs, **{k: v[0] for k, v in obs_p.items()}}, make_torch_gen(0, "policy"),
                           deterministic=True).action["selected"]
    assert set(inv[det].tolist()) == set(det_p.tolist())


@pytest.mark.parametrize("cls", [JointPolicy, StackedPolicy])
def test_sample_logp_matches_evaluate(cls):
    cfg = make_cfg()
    pol = make_policy(cls, cfg)
    obs = rand_obs(cfg, seed=2)
    with torch.no_grad():
        out = pol.sample(obs, make_torch_gen(1, "policy"))
        lp, _, va = pol.evaluate(batchify(obs), batch_action(out.action))
    assert list(out.logp) == list(pol.agent_names)
    for name in pol.agent_names:
        assert torch.allclose(out.logp[name].detach().reshape(-1), lp[name].reshape(-1), atol=1e-5)
        assert torch.allclose(out.value[name].detach().reshape(-1), va[name].reshape(-1), atol=1e-5)
        assert torch.isfinite(out.entropy[name]).all()


def test_unchosen_clients_do_not_count_in_logp():
    cfg = make_cfg()
    pol = make_policy(JointPolicy, cfg)
    obs = rand_obs(cfg, seed=3)
    with torch.no_grad():
        out = pol.sample(obs, make_torch_gen(2, "policy"))
        obs_b, act_b = batchify(obs), batch_action(out.action)
        lp0, en0, _ = pol.evaluate(obs_b, act_b)
        chosen = out.action["selected"].tolist()
        unchosen = next(i for i in range(cfg.fl.N) if i not in chosen)
        mod = {k: v.clone() for k, v in act_b.items()}
        for key in ("lr_idx", "epochs_idx", "batch_idx", "alpha_idx"):
            mod[key][0, unchosen] = (mod[key][0, unchosen] + 1) % 3
        lp1, en1, _ = pol.evaluate(obs_b, mod)
        assert torch.allclose(lp0["joint"], lp1["joint"], atol=1e-6)
        assert torch.allclose(en0["joint"], en1["joint"], atol=1e-6)
        mod2 = {k: v.clone() for k, v in act_b.items()}
        mod2["alpha_idx"][0, chosen[0]] = (mod2["alpha_idx"][0, chosen[0]] + 2) % 5
        lp2, _, _ = pol.evaluate(obs_b, mod2)
    assert not torch.allclose(lp0["joint"], lp2["joint"], atol=1e-7)


@pytest.mark.parametrize("cls", [JointPolicy, StackedPolicy])
def test_masked_clients_never_selected(cls):
    cfg = make_cfg()
    pol = make_policy(cls, cfg)
    elig = torch.tensor([True, False, True, True, False, True])
    emask = torch.ones(cfg.fl.N, 3, dtype=torch.bool)
    emask[:, 2] = False
    emask[3, 1] = False
    obs = rand_obs(cfg, seed=4, eligible=elig, emask=emask)
    gen = make_torch_gen(3, "policy")
    with torch.no_grad():
        for i in range(60):
            out = pol.sample(obs, gen, deterministic=(i == 0))
            sel = out.action["selected"]
            assert len(sel) == cfg.fl.k == len(set(sel.tolist())) and elig[sel].all()
            assert emask[sel, out.action["epochs_idx"][sel]].all()
            for name in pol.agent_names:
                assert torch.isfinite(out.logp[name]).all() and torch.isfinite(out.entropy[name]).all()


@pytest.mark.parametrize("cls", [JointPolicy, StackedPolicy])
def test_fewer_eligible_than_k(cls):
    cfg = make_cfg()
    pol = make_policy(cls, cfg)
    elig = torch.zeros(cfg.fl.N, dtype=torch.bool)
    elig[4] = True
    with torch.no_grad():
        out = pol.sample(rand_obs(cfg, eligible=elig), make_torch_gen(0, "policy"))
    assert out.action["selected"].tolist() == [4]


# ----------------------------------------------------------------------------- environment
def test_env_reset_step_shapes(bundle, pools):
    cfg = make_cfg()
    env = make_env(cfg, bundle, pools)
    n, k, t_total, d = cfg.fl.N, cfg.fl.k, cfg.fl.T, state_dim(cfg)
    assert (env.N, env.k, env.T, env.state_dim) == (n, k, t_total, d)
    obs = env.reset(SPEC_DRIFT)
    assert obs["state"].shape == (n, d) and obs["eligible"].dtype == torch.bool and obs["eligible"].shape == (n,)
    assert obs["epochs_mask"].shape == (n, 3) and isinstance(obs["round"], int)
    done, steps = False, 0
    while not done:
        n_elig = int(obs["eligible"].sum())
        obs, reward, done, info = env.step(default_action(obs, cfg))
        steps += 1
        assert isinstance(reward, (float, int)) and np.isfinite(reward) and obs["state"].shape == (n, d)
        assert all(key in info for key in INFO_KEYS), [key for key in INFO_KEYS if key not in info]
        assert len(info["selected"]) == min(k, n_elig) and info["rounds_since_local"].shape == (n,)
    assert steps == t_total
    cfg_w = make_cfg(["state.memory_mode=gru_window"])
    env_w = make_env(cfg_w, bundle, pools)
    win = env_w.reset(SPEC)["window"]
    assert win.shape == (n, cfg_w.state.gru_window, state_dim(cfg_w))


def test_env_probe_count_increases_on_drift(bundle, pools):
    cfg = make_cfg()
    counts = {}
    for spec in (SPEC, SPEC_DRIFT):
        env = make_env(cfg, bundle, pools)
        obs, done = env.reset(spec), False
        while not done:
            obs, _, done, info = env.step(default_action(obs, cfg))
        counts[spec.drift] = info["n_probes"]
    assert counts["abrupt"] == counts["none"] + 1


def test_env_reward_composition(bundle, pools):
    cfg = make_cfg()
    w = cfg.reward
    env = make_env(cfg, bundle, pools)
    obs, done = env.reset(SPEC_DRIFT), False
    while not done:
        obs, reward, done, info = env.step(default_action(obs, cfg, alpha_value=0.5))
        base = (w.w_global * info["r_global"] + w.w_pers * info["r_pers"] + w.w_fair * info["r_fair"]
                - w.w_comm * info["comm_norm"] - w.w_comp * info["comp_norm"])
        bonus = info["final_bonus"]
        assert abs(reward - (base + bonus)) < 1e-4
        assert 0.0 <= info["comm_norm"] <= w.cost_clip and 0.0 <= info["comp_norm"] <= w.cost_clip
        assert abs(info["mean_alpha"] - 0.5) < 1e-6
        if done:
            assert abs(bonus - w.final_bonus * info["global_val_acc"]) < 1e-5
        else:
            assert bonus == 0.0


def test_alpha_zero_gives_zero_personalization_gain(bundle, pools):
    cfg = make_cfg()
    env = make_env(cfg, bundle, pools)
    obs, done = env.reset(SPEC_DRIFT), False
    while not done:
        obs, _, done, info = env.step(default_action(obs, cfg, alpha_value=0.0))
        assert info["pers_gain_chosen"] == 0.0 and info["r_pers"] == 0.0 and info["pers_gain_all"] == 0.0
        assert info["mean_alpha"] == 0.0


def test_env_determinism(bundle, pools):
    cfg = make_cfg()
    env = make_env(cfg, bundle, pools)
    runs = []
    for _ in range(2):
        obs = env.reset(SPEC)
        act = default_action(obs, cfg)
        _, reward, _, _ = env.step(act)
        runs.append((obs["state"].clone(), reward))
    assert torch.equal(runs[0][0], runs[1][0]) and abs(runs[0][1] - runs[1][1]) < 1e-6

def test_baselines_respect_masks(bundle, pools):
    base = make_cfg(LOW_BATTERY)
    for name in [m for m in all_method_names() if not is_rl(m)]:
        cfg = apply_method(base, name)
        env = make_env(cfg, bundle, pools)
        try:
            ctrl = build_controller(name, cfg, env.state_dim, None)
        except Exception:  # noqa: BLE001
            if name.startswith("tuned_"):  # needs tuning.json from scripts/tune_baselines.py
                continue
            raise
        obs, done = env.reset(SPEC_DRIFT), False
        ctrl.seed = 0
        ctrl.reset(obs)
        ctrl.bind_env(env, data=(bundle.train_x, bundle.train_y))  # AFTER reset
        rng = make_rng(0, "eval_select")
        while not done:
            act = ctrl.act(obs, rng, True)
            sel = act["selected"]
            assert len(sel) == len(set(sel.tolist())) == min(cfg.fl.k, int(obs["eligible"].sum())), name
            assert obs["eligible"][sel].all(), f"{name} picked an ineligible client"
            assert obs["epochs_mask"][sel, act["epochs_idx"][sel]].all(), f"{name} used a masked epochs option"
            nxt, reward, done, info = env.step(act)
            ctrl.observe(obs, act, reward, nxt, info)
            obs = nxt

# ----------------------------------------------------------------------------- PPO
def test_ppo_learns_toy_contextual_bandit():
    cfg = make_cfg(["policy.active_heads.train=false", "policy.active_heads.mix=false", "ppo.lr=0.005",
                    "ppo.epochs=4", "ppo.minibatches=4"])
    torch.manual_seed(0)
    pol = JointPolicy(cfg, state_dim(cfg))
    learner = PPOLearner(list(pol.parameters()), cfg, "joint")
    buf = RolloutBuffer(cfg, list(pol.agent_names))
    n, d = cfg.fl.N, state_dim(cfg)
    g, pgen, rng = torch.Generator().manual_seed(0), make_torch_gen(0, "policy"), np.random.default_rng(0)
    means = []
    for _ in range(80):
        rewards = []
        for _ in range(32):
            state = torch.randn(n, d, generator=g)
            obs = {"state": state, "eligible": torch.ones(n, dtype=torch.bool),
                   "epochs_mask": torch.ones(n, 3, dtype=torch.bool), "round": 0}
            with torch.no_grad():
                out = pol.sample(obs, pgen)
            reward = float((state[out.action["selected"], 0] > 0).sum())  # random policy ~ 1.0, best ~ 2.0
            rewards.append(reward)
            buf.add(obs, out, {"joint": reward}, True)
            buf.finish_episode(last_value=0.0)
        assert len(buf) > 0
        def _ev(ob, ac):
            lp, en, va = pol.evaluate(ob, ac)
            return lp["joint"], en["joint"], va["joint"]
        stats = learner.update(buf, _ev, rng)
        assert isinstance(stats, dict) and all(np.isfinite(float(v)) for v in stats.values())
        buf.clear()
        assert len(buf) == 0
        means.append(np.mean(rewards))
    first, last = float(np.mean(means[:10])), float(np.mean(means[-10:]))
    assert last > 1.35 and last > first + 0.25, f"no learning: first={first:.2f} last={last:.2f} (random ~ 1.0)"
    learner.load_state_dict(learner.state_dict())

# ----------------------------------------------------------------------------- runs
def test_resume_equals_uninterrupted(tmp_path):
    ra, rb = tmp_path / "a", tmp_path / "b"
    assert config_hash(make_cfg(["ppo.episodes=3"], ra)) == config_hash(make_cfg(["ppo.episodes=6"], ra))
    with install_synthetic_data(), block_official_test():
        da = Path(train(make_cfg(["ppo.episodes=6"], ra), "joint", "default", 0))
        train(make_cfg(["ppo.episodes=3"], rb), "joint", "default", 0)
        db = Path(train(make_cfg(["ppo.episodes=6"], rb), "joint", "default", 0))  # restart = resume
    ca, cb = CheckpointManager(da).load_latest(), CheckpointManager(db).load_latest()
    assert ca is not None and cb is not None
    n_compared, problems = compare_checkpoints(ca, cb)
    assert n_compared > 0 and not problems, problems
    assert not compare_run_logs(da, db)


def test_corrupt_latest_checkpoint_falls_back(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    cm = CheckpointManager(rd)
    assert cm.load_latest() is None
    for ep in (1, 2, 3):
        cm.save({"marker": ep, "w": torch.full((3,), float(ep))}, episode=ep, is_best=(ep == 2))
    assert cm.load_latest()["marker"] == 3
    files = [p for p in rd.rglob("*") if p.is_file() and p.suffix in (".pt", ".pth", ".ckpt", ".pkl")
             and "best" not in p.name.lower()]
    newest = [p for p in files if "latest" in p.name.lower() or 3 in [int(x) for x in re.findall(r"\d+", p.name)]]
    assert newest, f"could not identify the latest checkpoint file among {[p.name for p in files]}"
    for p in newest:
        p.write_bytes(b"corrupt")
    assert CheckpointManager(rd).load_latest()["marker"] == 2
    assert CheckpointManager(rd).load_best()["marker"] == 2
    assert not [p for p in rd.rglob("*") if p.is_file() and "tmp" in p.suffix.lower()]


def test_config_hash_ignores_non_result_keys(tmp_path):
    base = config_hash(make_cfg())
    assert config_hash(make_cfg(["ppo.episodes=999"])) == base
    assert config_hash(make_cfg(["logging.level=DEBUG"])) == base
    assert config_hash(make_cfg([f"run.runs_dir={tmp_path}"])) == base
    assert config_hash(make_cfg(["fl.k=3"])) != base
    validate(make_cfg())
    with pytest.raises(Exception):
        bad = make_cfg(["ppo.update_every_episodes=2", "ppo.checkpoint_every=3"])
        validate(bad)
        check_schedule(bad)


# ----------------------------------------------------------------------------- leakage
def _py(*parts):
    base = ROOT.joinpath(*parts)
    return sorted(base.rglob("*.py")) if base.is_dir() else [base]

def _guard_ranges(tree):
    return [(n.lineno, n.end_lineno) for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "forbid_test_access"]


def _uses_official_test(tree):
    guard = _guard_ranges(tree)
    for node in ast.walk(tree):
        if any(a <= getattr(node, "lineno", 0) <= b for a, b in guard):
            continue
        if isinstance(node, ast.ImportFrom) and any(a.name == "get_official_test" for a in node.names):
            return True
        if isinstance(node, ast.Attribute) and node.attr == "get_official_test":
            return True
        if isinstance(node, ast.Name) and node.id == "get_official_test":
            return True
    return False


def test_training_code_has_no_test_set_access_static():
    allowed = {ROOT / "fedrl" / "data" / "scenario.py", ROOT / "fedrl" / "data" / "datasets.py"}
    eval_dir = ROOT / "fedrl" / "eval"
    offenders = []
    for path in _py("fedrl"):
        if eval_dir in path.parents or path in allowed:
            continue
        if _uses_official_test(ast.parse(path.read_text(encoding="utf-8"))):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"get_official_test used outside fedrl/eval and scenario.py: {offenders}"
    pattern = re.compile(r"scenarios\s*(\.|\[\s*['\"])\s*test|test_scenarios\s*\(")
    bad = []
    for sub in ("rl", "runs", "env", "controllers"):
        for p in _py("fedrl", sub):
            text = p.read_text(encoding="utf-8")
            guard = _guard_ranges(ast.parse(text))
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line) and not any(a <= i <= b for a, b in guard):
                    bad.append(f"{p.relative_to(ROOT)}:{i}")
    assert not bad, f"test scenarios referenced in training-side code: {bad}"

def test_training_does_not_touch_test_data_dynamic(trained):
    cfg, run_dir = trained  # the fixture trained with get_official_test patched to raise
    assert run_dir.is_dir() and CheckpointManager(run_dir).load_latest() is not None


# ----------------------------------------------------------------------------- end to end
def test_tiny_end_to_end_smoke(trained, bundle, pools):
    cfg, run_dir = trained
    spec = ScenarioSpec(0.2, "abrupt", 1000)
    for name in ("fedavg", "joint"):
        cfg_m = apply_method(cfg, name)
        ctrl = build_controller(name, cfg_m, state_dim(cfg_m), run_dir=run_dir if is_rl(name) else None)
        res = evaluate_controller(ctrl, spec, 1000, cfg_m, bundle, pools, DEV)
        assert isinstance(res, dict) and res
        json.dumps(to_jsonable(res))
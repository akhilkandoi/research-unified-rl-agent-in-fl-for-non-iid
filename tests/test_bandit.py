"""LinUCB toy-problem, masking and state round-trip tests."""
import numpy as np
import torch

from fedrl.config import load_config
from fedrl.controllers.bandit import LinUCB
from fedrl.controllers.base import synthetic_obs
from fedrl.env.state import state_dim
from fedrl.utils.io import atomic_torch_save

CFG = load_config("configs/smoke.yaml")
D = state_dim(CFG)


def _train(bandit, rng, rounds):
    for _ in range(rounds):
        obs = synthetic_obs(CFG, D, rng)
        act = bandit.act(obs, rng, deterministic=False)
        reward = float(obs["state"][act["selected"], 0].mean()) + 0.1 * rng.normal()
        bandit.observe(obs, act, reward, obs, {})


def test_bandit_learns_toy_problem():
    rng = np.random.default_rng(0)
    b = LinUCB(CFG, D)
    _train(b, rng, 400)
    assert int(np.argmax(b.theta())) == 0
    picked = []
    for _ in range(200):
        obs = synthetic_obs(CFG, D, rng)
        act = b.act(obs, rng, deterministic=True)
        picked.append(float(obs["state"][act["selected"], 0].mean()))
    assert np.mean(picked) > 0.3  # random selection averages about 0.0


def test_bandit_never_picks_ineligible():
    rng = np.random.default_rng(1)
    b = LinUCB(CFG, D)
    for _ in range(50):
        obs = synthetic_obs(CFG, D, rng, n_ineligible=3)
        act = b.act(obs, rng, deterministic=False)
        elig = obs["eligible"].cpu().numpy()
        assert all(elig[i] for i in act["selected"].tolist())
        b.observe(obs, act, 1.0, obs, {})


def test_bandit_state_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    b = LinUCB(CFG, D)
    _train(b, rng, 30)
    path = tmp_path / "bandit.pt"
    atomic_torch_save(b.state_dict(), path)
    b2 = LinUCB(CFG, D)
    b2.load_state_dict(torch.load(path, weights_only=False))
    assert np.allclose(b.A, b2.A) and np.allclose(b.b, b2.b) and b.n_updates == b2.n_updates
    obs = synthetic_obs(CFG, D, np.random.default_rng(5))
    a1 = b.act(obs, np.random.default_rng(9), deterministic=True)
    a2 = b2.act(obs, np.random.default_rng(9), deterministic=True)
    assert torch.equal(a1["selected"], a2["selected"])


def test_bandit_rejects_wrong_shape_state():
    b = LinUCB(CFG, D)
    bad = {"A": np.eye(D + 1), "b": np.zeros(D + 1), "n_updates": 0}
    try:
        b.load_state_dict(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
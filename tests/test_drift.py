import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
from helpers_data import make_world

from fedrl.data.drift import apply_drift, drift_due, gradual_lambda, proportions_at
from fedrl.data.partition import draw_client_data
from fedrl.data.scenario import Scenario
from fedrl.data.scenario_sets import ScenarioSpec


@pytest.fixture(scope="module")
def world():
    return make_world()


def test_drift_due_table(world):
    cfg, _, _ = world
    assert not any(drift_due("none", t, cfg) for t in range(1, 31))
    assert [t for t in range(1, 31) if drift_due("abrupt", t, cfg)] == [15]
    assert [t for t in range(1, 31) if drift_due("gradual", t, cfg)] == list(range(15, 23))
    with pytest.raises(ValueError):
        drift_due("sideways", 1, cfg)


def test_gradual_lambda_endpoints_and_monotonic(world):
    cfg, _, _ = world
    assert gradual_lambda(14, cfg) == 0.0
    assert gradual_lambda(15, cfg) == pytest.approx(1 / 8)
    assert gradual_lambda(22, cfg) == pytest.approx(1.0)
    assert gradual_lambda(23, cfg) == 1.0
    lam = [gradual_lambda(t, cfg) for t in range(1, 31)]
    assert all(b >= a for a, b in zip(lam, lam[1:]))
    assert all(0.0 <= x <= 1.0 for x in lam)


def test_abrupt_changes_exactly_at_round_15(world):
    cfg, bundle, pools = world
    sc = Scenario(ScenarioSpec(0.3, "abrupt", 3), bundle, pools, cfg)
    p0 = sc.proportions_matrix().copy()
    tr0 = [c.train_idx.clone() for c in sc.clients]
    changed = []
    for t in range(1, 31):
        changed.append(sc.apply_drift(t))
        if t < 15:
            np.testing.assert_allclose(sc.proportions_matrix(), p0)
            assert all((c.train_idx == a).all() for c, a in zip(sc.clients, tr0))
    assert [t for t, c in enumerate(changed, 1) if c] == [15]
    assert not np.allclose(sc.proportions_matrix(), p0)
    np.testing.assert_allclose(sc.proportions_matrix(), sc.target_proportions)
    assert not sc.apply_drift(15)  # already applied


def test_none_drift_never_changes(world):
    cfg, bundle, pools = world
    sc = Scenario(ScenarioSpec(0.3, "none", 3), bundle, pools, cfg)
    assert not any(sc.apply_drift(t) for t in range(1, 31))


def test_gradual_scenario_follows_formula(world):
    cfg, bundle, pools = world
    sc = Scenario(ScenarioSpec(0.3, "gradual", 4), bundle, pools, cfg)
    p0, pn = sc.base_proportions.copy(), sc.target_proportions.copy()
    changed, dist = [], []
    for t in range(1, 31):
        changed.append(sc.apply_drift(t))
        lam = gradual_lambda(t, cfg)
        np.testing.assert_allclose(sc.proportions_matrix(), (1 - lam) * p0 + lam * pn, atol=1e-12)
        dist.append(np.abs(sc.proportions_matrix() - p0).sum())
    assert [t for t, c in enumerate(changed, 1) if c] == list(range(15, 23))
    assert all(b >= a - 1e-12 for a, b in zip(dist, dist[1:]))


def test_pure_apply_drift_is_deterministic(world):
    cfg, bundle, pools = world
    sc = Scenario(ScenarioSpec(0.3, "abrupt", 5), bundle, pools, cfg)
    args = ("abrupt", 15, 5, sc.base_proportions, sc.target_proportions, sc.sizes, pools, cfg)
    u1, u2 = apply_drift(*args), apply_drift(*args)
    assert u1.changed and u2.changed
    assert all((a.train_idx == b.train_idx).all() for a, b in zip(u1.clients, u2.clients))
    np.testing.assert_allclose(u1.proportions, proportions_at("abrupt", 15, sc.base_proportions,
                                                              sc.target_proportions, cfg))
    assert not apply_drift("abrupt", 14, 5, sc.base_proportions, sc.target_proportions, sc.sizes, pools, cfg).changed


def test_resampling_uses_new_proportions_pure(world):
    cfg, bundle, pools = world
    y = bundle.train_y.numpy()
    p = np.zeros(10)
    p[7] = 1.0
    d = draw_client_data(1, 0, 300, p, pools, cfg, tag="drift_r15")
    assert (y[d.train_idx] == 7).all() and (y[d.val_idx] == 7).all()


def test_resampling_uses_new_proportions_scenario(world):
    cfg, bundle, pools = world
    y = bundle.train_y.numpy()
    sc = Scenario(ScenarioSpec(0.3, "abrupt", 6), bundle, pools, cfg)
    old = sc.proportions_matrix().copy()
    for t in range(1, 16):
        sc.apply_drift(t)
    new = sc.proportions_matrix()
    emp = np.stack([np.bincount(y[c.train_idx.numpy()], minlength=10) / c.n for c in sc.clients])
    l1_new = np.abs(emp - new).sum(1).mean()
    l1_old = np.abs(emp - old).sum(1).mean()
    assert l1_new < 0.25 and l1_new < l1_old
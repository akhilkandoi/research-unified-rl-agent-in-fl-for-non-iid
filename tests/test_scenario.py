import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
import torch
from helpers_data import make_world

from fedrl.data.partition import val_size
from fedrl.data.scenario import Scenario, sample_resources
from fedrl.data.scenario_sets import ScenarioSpec
from fedrl.utils.seeding import make_rng


@pytest.fixture(scope="module")
def world():
    return make_world()


def same(a: Scenario, b: Scenario) -> bool:
    if a.init_seed != b.init_seed or a.N != b.N:
        return False
    for x, y in zip(a.clients, b.clients):
        if not (torch.equal(x.train_idx, y.train_idx) and torch.equal(x.val_idx, y.val_idx)
                and torch.equal(x.personal_val_idx, y.personal_val_idx)
                and np.array_equal(x.proportions, y.proportions)
                and (x.compute_speed, x.bandwidth, x.battery0) == (y.compute_speed, y.bandwidth, y.battery0)):
            return False
    return True


def test_same_seed_identical_different_seed_differs(world):
    cfg, bundle, pools = world
    a = Scenario(ScenarioSpec(0.2, "abrupt", 1000), bundle, pools, cfg)
    b = Scenario(ScenarioSpec(0.2, "abrupt", 1000), bundle, pools, cfg)
    c = Scenario(ScenarioSpec(0.2, "abrupt", 1001), bundle, pools, cfg)
    assert same(a, b)
    assert not same(a, c)
    assert not np.allclose(a.proportions_matrix(), c.proportions_matrix())
    assert a.init_seed != c.init_seed


def test_client_sizes_and_val_sets(world):
    cfg, bundle, pools = world
    sc = Scenario(ScenarioSpec(0.3, "none", 7), bundle, pools, cfg)
    assert len(sc.clients) == sc.N == 20
    gv = set(pools.global_val_np.tolist())
    for c in sc.clients:
        assert 150 <= c.n <= 2000
        assert len(np.unique(c.train_idx.numpy())) == c.n
        assert len(c.val_idx) == val_size(c.n, cfg)
        assert len(c.personal_val_idx) == len(c.val_idx)
        n_glob = sum(int(i) in gv for i in c.personal_val_idx.tolist())
        assert n_glob == len(c.val_idx) - int(round(0.8 * len(c.val_idx)))
        assert c.compute_speed > 0 and c.bandwidth > 0 and 0.5 <= c.battery0 <= 1.0
        assert c.proportions.shape == (10,) and abs(c.proportions.sum() - 1) < 1e-9


def test_personal_test_sets_identical_across_scenarios(world):
    cfg, bundle, pools = world
    spec = ScenarioSpec(0.2, "abrupt", 1002)
    a, b = Scenario(spec, bundle, pools, cfg), Scenario(spec, bundle, pools, cfg)
    for i in range(a.N):
        ta, tb = a.personal_test_idx(i), b.personal_test_idx(i)
        assert torch.equal(ta, tb) and len(ta) == 200 and len(torch.unique(ta)) == 200
        assert int(ta.max()) < len(bundle.test_y)
    for t in range(1, 31):
        a.apply_drift(t)
        b.apply_drift(t)
    assert all(torch.equal(a.personal_test_idx(i), b.personal_test_idx(i)) for i in range(a.N))
    assert not torch.equal(a.personal_test_idx(0), a.personal_test_idx(1))


def test_personal_test_follows_current_proportions(world):
    cfg, bundle, pools = world
    sc = Scenario(ScenarioSpec(0.3, "abrupt", 1003), bundle, pools, cfg)
    for t in range(1, 16):
        sc.apply_drift(t)
    y = bundle.test_y.numpy()
    l1 = [np.abs(np.bincount(y[sc.personal_test_idx(i).numpy()], minlength=10) / 200
                 - sc.clients[i].proportions).sum() for i in range(sc.N)]
    assert np.mean(l1) < 0.3


def test_resources_lognormal_median_one(world):
    cfg, _, _ = world
    speed, bw, b0 = sample_resources(make_rng(0, "r"), 20000, cfg)
    for v in (speed, bw):
        assert 0.97 < np.median(v) < 1.03
        assert abs(np.log(v).std() - 0.5) < 0.02
    assert b0.min() >= 0.5 and b0.max() <= 1.0 and abs(b0.mean() - 0.75) < 0.01
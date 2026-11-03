import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
from helpers_data import make_cfg, make_synthetic_bundle, make_world

from fedrl.data.pools import Pools, assert_pools_disjoint, build_pools


@pytest.fixture(scope="module")
def world():
    return make_world()


def test_pools_disjoint_and_global_val_balanced(world):
    cfg, bundle, pools = world
    y = bundle.train_y.numpy()
    gv = pools.global_val_idx.numpy()
    assert len(gv) == 500
    assert (np.bincount(y[gv], minlength=10) == 50).all()
    tr = np.concatenate([t.numpy() for t in pools.train_by_class])
    va = np.concatenate([t.numpy() for t in pools.val_by_class])
    assert not np.intersect1d(gv, tr).size
    assert not np.intersect1d(gv, va).size
    assert not np.intersect1d(tr, va).size
    assert len(gv) + len(tr) + len(va) == len(y)
    assert abs(len(tr) / (len(tr) + len(va)) - 0.8) < 0.01
    for c in range(10):
        assert (y[pools.train_by_class[c].numpy()] == c).all()
        assert (y[pools.val_by_class[c].numpy()] == c).all()
    assert_pools_disjoint(pools)


def test_disjoint_check_detects_overlap(world):
    _, _, pools = world
    bad = Pools(global_val_idx=pools.train_by_class[0][:5].clone(),
                train_by_class=pools.train_by_class, val_by_class=pools.val_by_class)
    with pytest.raises(AssertionError):
        assert_pools_disjoint(bad)


def test_pool_seed_determinism(world):
    cfg, bundle, pools = world
    again = build_pools(bundle, cfg)
    assert (again.global_val_idx == pools.global_val_idx).all()
    other = build_pools(bundle, make_cfg(["data.pool_seed=1"]))
    assert not (other.global_val_idx == pools.global_val_idx).all()


def test_too_small_class_raises():
    cfg = make_cfg(["data.global_val_size=5000"])
    with pytest.raises(ValueError):
        build_pools(make_synthetic_bundle(n_train=3000), cfg)
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
from helpers_data import make_cfg, make_world

from fedrl.data import partition
from fedrl.data.partition import (build_personal_val, cap_and_redistribute, draw_counts, draw_indices,
                                  sample_client_sizes, sample_proportions, val_size)
from fedrl.utils.seeding import make_rng


@pytest.fixture(scope="module")
def world():
    return make_world()


def test_sizes_within_bounds():
    cfg = make_cfg()
    s = sample_client_sizes(make_rng(0, "t"), 5000, cfg)
    assert np.issubdtype(s.dtype, np.integer)
    assert s.min() >= 150 and s.max() <= 2000
    assert 500 < s.mean() < 700


def test_proportions_follow_dirichlet():
    a, C, n = 0.3, 10, 20000
    P = sample_proportions(make_rng(1, "t"), a, C, n)
    assert P.shape == (n, C)
    np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-9)
    assert np.abs(P.mean(0) - 1 / C).max() < 0.005
    var_th = (1 / C) * (1 - 1 / C) / (a * C + 1)
    np.testing.assert_allclose(P.var(0), var_th, rtol=0.15)
    assert sample_proportions(make_rng(2, "t"), 0.05, C).shape == (C,)


def test_tiny_a_is_finite():
    P = sample_proportions(make_rng(3, "t"), 0.01, 10, 500)
    assert np.all(np.isfinite(P)) and np.allclose(P.sum(1), 1.0)


def test_counts_sum_and_follow_proportions():
    p = np.array([0.5, 0.3, 0.2] + [0.0] * 7)
    rng = make_rng(4, "t")
    C = np.stack([draw_counts(rng, 400, p) for _ in range(5000)])
    assert (C.sum(1) == 400).all()
    assert np.abs(C.mean(0) / 400 - p).max() < 0.01


def test_draw_indices_len_unique_and_cap():
    pools = [np.arange(c * 1000, (c + 1) * 1000) for c in range(10)]
    rng = make_rng(5, "t")
    for _ in range(100):
        n = int(rng.integers(150, 2001))
        p = sample_proportions(rng, 0.05, 10)
        idx = draw_indices(rng, n, p, pools)
        assert len(idx) == n and len(np.unique(idx)) == n
        assert (np.bincount(idx // 1000, minlength=10) <= 1000).all()


def test_cap_redistribute_warns():
    class Cap(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.WARNING)
            self.records = []

        def emit(self, record):
            self.records.append(record)

    h = Cap()
    partition.log.addHandler(h)
    try:
        pools = [np.arange(c * 10, (c + 1) * 10) for c in range(10)]
        p = np.zeros(10)
        p[0] = 1.0
        idx = draw_indices(make_rng(6, "t"), 25, p, pools)
    finally:
        partition.log.removeHandler(h)
    assert len(idx) == 25 and len(np.unique(idx)) == 25
    assert (idx // 10 == 0).sum() == 10
    assert any(r.levelno == logging.WARNING for r in h.records)


def test_cap_redistribute_direct():
    rng = make_rng(7, "t")
    out = cap_and_redistribute(rng, np.array([30, 0, 0]), np.array([1.0, 0.0, 0.0]), np.array([10, 10, 10]))
    assert out.sum() == 30 and out[0] == 10 and (out <= 10).all()


def test_personal_val_composition(world):
    cfg, bundle, pools = world
    rng = make_rng(8, "t")
    own = draw_indices(rng, 200, sample_proportions(rng, 0.3, 10), pools.val_np)
    out = build_personal_val(rng, own, pools.global_val_np, 0.8)
    n_glob = np.isin(out, pools.global_val_np).sum()
    assert len(out) == len(own) and n_glob == len(own) - int(round(0.8 * len(own)))
    assert np.isin(out[~np.isin(out, pools.global_val_np)], np.concatenate(pools.val_np)).all()
    assert len(np.unique(out)) == len(out)


def test_val_size_rule():
    cfg = make_cfg()
    assert val_size(150, cfg) == 50 and val_size(2000, cfg) == 500 and val_size(600, cfg) == 150
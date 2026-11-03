"""Tests for fedrl/env/state.py."""
import math
from pathlib import Path

import pytest
import torch

from fedrl.config import load_config
from fedrl.env.state import (BASE_FEATURE_NAMES, FEATURE_NAMES, build_state, feature_names,
                             state_dim)

ROOT = Path(__file__).resolve().parents[1]


def make_cfg(*overrides):
    return load_config(str(ROOT / "configs" / "default.yaml"), list(overrides))


@pytest.mark.parametrize("mode,base", [("none", 24), ("ema", 24), ("ema_delta", 26),
                                       ("gru_window", 24), ("gru_const", 24)])
@pytest.mark.parametrize("extra", [False, True])
def test_state_dim_every_mode(mode, base, extra):
    cfg = make_cfg(f"state.memory_mode={mode}", f"state.extra_features={'true' if extra else 'false'}")
    assert state_dim(cfg) == base + (5 if extra else 0)
    assert len(feature_names(cfg)) == state_dim(cfg)


def test_default_is_24():
    assert state_dim(make_cfg()) == 24 == len(BASE_FEATURE_NAMES)
    assert FEATURE_NAMES[:24] == BASE_FEATURE_NAMES


def _inputs(cfg, N=2, **over):
    d = dict(
        proportions=torch.eye(10)[[3, 4]][:N], n=torch.full((N,), 600.0),
        probe_loss=torch.full((N,), 1.5), probe_acc=torch.full((N,), 0.7),
        prev_loss=torch.full((N,), 2.0), prev_acc=torch.full((N,), 0.6),
        last_div=torch.full((N,), 1.0), compute_speed=torch.full((N,), math.e),
        bandwidth=torch.full((N,), math.exp(-1.0)), battery=torch.full((N,), 0.8),
        mem_features=torch.tensor([[0.1, 0.2, 0.3, 0.4]] * N), round_idx=15,
        mem_delta=torch.zeros(N, 2), extras=torch.zeros(N, 5),
    )
    d.update(over)
    return d


def test_feature_order_and_scaling():
    cfg = make_cfg()
    s = build_state(cfg, **_inputs(cfg))
    assert s.shape == (2, 24)
    col = lambda name: s[:, FEATURE_NAMES.index(name)]  # noqa: E731
    assert col("label_prop_3")[0] == pytest.approx(1.0) and col("label_prop_4")[1] == pytest.approx(1.0)
    assert col("log_n")[0] == pytest.approx(math.log(600) / math.log(2000))
    assert col("probe_loss")[0] == pytest.approx(0.5)
    assert col("probe_acc")[0] == pytest.approx(0.7)
    assert col("d_probe_loss")[0] == pytest.approx(-0.5 / 3.0)
    assert col("d_probe_acc")[0] == pytest.approx(0.1)
    assert col("last_div")[0] == pytest.approx(0.5)
    assert col("log_speed")[0] == pytest.approx(1.0 / 3.0)
    assert col("log_bandwidth")[0] == pytest.approx(-1.0 / 3.0)
    assert col("battery")[0] == pytest.approx(0.8)
    for name, v in zip(["mem_ema_loss", "mem_ema_div", "mem_idle", "mem_picked"], [0.1, 0.2, 0.3, 0.4]):
        assert col(name)[0] == pytest.approx(v)
    assert col("round_frac")[0] == pytest.approx(0.5)


def test_first_probe_deltas_zero():
    cfg = make_cfg()
    s = build_state(cfg, **_inputs(cfg, prev_loss=None, prev_acc=None))
    assert s[:, FEATURE_NAMES.index("d_probe_loss")].abs().max() == 0
    assert s[:, FEATURE_NAMES.index("d_probe_acc")].abs().max() == 0


def test_scaling_bounds_under_extreme_inputs():
    cfg = make_cfg()
    N = 3
    s = build_state(cfg, **_inputs(
        cfg, N=N, proportions=torch.tensor([[5.0] * 10, [0.0] * 9 + [1.0], [1.0] * 10]),
        n=torch.tensor([1.0, 5000.0, 600.0]), probe_loss=torch.tensor([100.0, 0.0, float("nan")]),
        probe_acc=torch.tensor([1.5, -0.2, 0.5]), prev_loss=torch.tensor([0.0, 100.0, 1.0]),
        prev_acc=torch.tensor([0.0, 1.0, 0.5]), last_div=torch.tensor([50.0, 0.0, 1.0]),
        compute_speed=torch.tensor([1e-9, 1e9, 1.0]), bandwidth=torch.tensor([1e9, 1e-9, 1.0]),
        battery=torch.tensor([1.5, -0.1, 0.5]), mem_features=torch.rand(N, 4), round_idx=99))
    assert torch.isfinite(s).all()
    rng = {"label_prop": (0, 1), "log_n": (0, 1), "probe_loss": (0, 2), "probe_acc": (0, 1),
           "d_probe_loss": (-1, 1), "d_probe_acc": (-1, 1), "last_div": (0, 1), "log_speed": (-1, 1),
           "log_bandwidth": (-1, 1), "battery": (0, 1), "mem_ema_loss": (0, 1), "mem_ema_div": (0, 1),
           "mem_idle": (0, 1), "mem_picked": (0, 1), "round_frac": (0, 1)}
    for j, name in enumerate(FEATURE_NAMES[:24]):
        key = "label_prop" if name.startswith("label_prop") else name
        lo, hi = rng[key]
        assert s[:, j].min() >= lo - 1e-6 and s[:, j].max() <= hi + 1e-6, name
    assert s[:, :10].sum(1).allclose(torch.ones(N))


def test_none_mode_zeroes_memory_block():
    cfg = make_cfg("state.memory_mode=none")
    s = build_state(cfg, **_inputs(cfg))
    assert s[:, 19:23].abs().max() == 0
    assert s.shape[1] == 24


def test_ema_delta_and_extras_columns():
    cfg = make_cfg("state.memory_mode=ema_delta", "state.extra_features=true")
    s = build_state(cfg, **_inputs(cfg, mem_delta=torch.full((2, 2), 0.25),
                                   extras=torch.arange(10.0).reshape(2, 5) / 10))
    assert s.shape == (2, 31)
    assert s[0, FEATURE_NAMES.index("mem_d_ema_loss")] == pytest.approx(0.25)
    assert s[0, FEATURE_NAMES.index("x_global_acc")] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        build_state(cfg, **_inputs(cfg, mem_delta=None))
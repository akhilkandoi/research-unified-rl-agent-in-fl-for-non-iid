"""Tests for fedrl/env/memory.py."""
from pathlib import Path

import pytest
import torch

from fedrl.config import load_config
from fedrl.env.memory import Memory
from fedrl.env.state import state_dim

ROOT = Path(__file__).resolve().parents[1]


def make_cfg(*overrides):
    return load_config(str(ROOT / "configs" / "default.yaml"), list(overrides))


def test_ema_loss_by_hand():
    cfg = make_cfg()  # beta 0.9, loss_scale 3, T 30
    m = Memory(cfg, 3)
    acc = torch.zeros(3)
    m.observe_probe(torch.tensor([3.0, 0.6, 6.0]), acc)
    assert torch.allclose(m.features()[:, 0], torch.tensor([1.0, 0.2, 2.0]))  # initialised at first probe
    m.observe_probe(torch.tensor([1.5, 0.6, 0.0]), acc)
    expected = 0.9 * torch.tensor([1.0, 0.2, 2.0]) + 0.1 * torch.tensor([0.5, 0.2, 0.0])
    assert torch.allclose(m.features()[:, 0], expected)


def test_div_ema_only_for_picked_and_counters():
    cfg = make_cfg()  # div_scale 2
    m = Memory(cfg, 3)
    m.observe_probe(torch.ones(3), torch.zeros(3))
    m.record_round([1], torch.tensor([0.5, 1.0, 0.0]))
    f = m.features()
    assert f[0, 1] == 0 and f[2, 1] == 0
    assert f[1, 1] == pytest.approx(0.1 * 0.5)
    m.record_round([1], torch.tensor([0.0, 0.2, 0.0]))
    assert m.features()[1, 1] == pytest.approx(0.9 * 0.05 + 0.1 * 0.1)
    assert torch.allclose(m.features()[:, 2], torch.tensor([2.0, 0.0, 2.0]) / 30)
    assert torch.allclose(m.features()[:, 3], torch.tensor([0.0, 2.0, 0.0]) / 30)
    assert m.last_div[1] == pytest.approx(0.2) and m.last_div[0] == 0


def test_none_mode_zero_features_but_counters_kept():
    m = Memory(make_cfg("state.memory_mode=none"), 3)
    m.observe_probe(torch.ones(3), torch.zeros(3))
    m.record_round([0], torch.ones(3))
    assert m.features().abs().max() == 0
    assert m.picked[0] == 1 and m.idle[1] == 1


def test_ema_delta_features():
    m = Memory(make_cfg("state.memory_mode=ema_delta"), 3)
    acc = torch.zeros(3)
    m.observe_probe(torch.tensor([3.0, 0.6, 6.0]), acc)
    assert m.delta_features().abs().max() == 0
    m.observe_probe(torch.tensor([1.5, 0.6, 0.0]), acc)
    assert torch.allclose(m.delta_features()[:, 0], torch.tensor([-0.05, 0.0, -0.2]), atol=1e-6)
    m.record_round([1], torch.tensor([0.0, 2.0, 0.0]))
    assert m.delta_features()[1, 1] == pytest.approx(0.1)


def test_gru_window_shape_and_padding():
    cfg = make_cfg("state.memory_mode=gru_window")
    D = state_dim(cfg)
    m = Memory(cfg, 3)
    assert m.window().shape == (3, 5, D) and m.window().abs().max() == 0
    for v in range(1, 7):
        m.push_state(torch.full((3, D), float(v)))
        if v == 2:
            w = m.window()
            assert w[:, :3].abs().max() == 0 and w[0, 3, 0] == 1 and w[0, 4, 0] == 2
    assert [float(x) for x in m.window()[0, :, 0]] == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert Memory(make_cfg("state.memory_mode=gru_const"), 3).window() is None
    assert Memory(make_cfg(), 3).window() is None


def test_extra_features_and_reset():
    cfg = make_cfg("state.extra_features=true")
    m = Memory(cfg, 2)
    m.set_reference(global_acc=0.3, comm_budget=10.0)
    m.observe_probe(torch.ones(2), torch.zeros(2))
    m.record_round([0], torch.ones(2), update_cos=torch.tensor([0.5]), global_acc=0.4,
                   global_acc_change=0.1, comm=2.0)
    e = m.extra_features()
    assert e.shape == (2, 5)
    assert e[0, 0] == pytest.approx(0.5) and e[1, 0] == 0
    assert e[0, 2] == pytest.approx(0.4) and e[0, 3] == pytest.approx(0.1) and e[0, 4] == pytest.approx(0.2)
    m.reset()
    assert m.extra_features()[:, 4].abs().max() == 0 and m.picked.sum() == 0 and not m.has_probe
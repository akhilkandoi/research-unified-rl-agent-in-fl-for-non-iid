from pathlib import Path

import numpy as np
import pytest
import torch

from fedrl.config import load_config
from fedrl.fl.costs import cost_reference, n_transfers, normalize_costs, round_costs
from fedrl.fl.resources import Battery, drain, eligible_mask, epochs_mask

ROOT = Path(__file__).resolve().parents[1]


def make_cfg(overrides=None):
    return load_config(str(ROOT / "configs" / "default.yaml"), overrides=list(overrides or []))


def test_drain_formula():
    cfg = make_cfg()
    assert drain(2, 600, cfg) == pytest.approx(0.05)
    assert drain(5, 300, cfg) == pytest.approx(0.05 / 1200 * 5 * 300)


def test_eligible_mask():
    cfg = make_cfg()
    n = torch.full((4,), 600.0)
    bat = torch.tensor([0.07, 0.08, 0.5, 0.0])  # threshold is 0.05 + drain(1,600)=0.075
    assert eligible_mask(bat, n, cfg).tolist() == [False, True, True, False]


def test_epochs_mask_and_consistency():
    cfg = make_cfg()
    n = torch.full((4,), 600.0)
    bat = torch.tensor([0.15, 0.12, 0.09, 0.07])
    m = epochs_mask(bat, n, cfg)
    assert m.tolist() == [[True, True, False], [True, True, False],
                          [True, False, False], [False, False, False]]
    assert torch.equal(m[:, 0], eligible_mask(bat, n, cfg))


def test_battery_drain_recharge_cap():
    cfg = make_cfg()
    n = torch.full((3,), 600.0)
    b = Battery(torch.tensor([0.9, 0.5, 1.0]), n, cfg)
    b.apply_drain([0], [2])
    assert b.level[0].item() == pytest.approx(0.85)
    assert b.level[1].item() == pytest.approx(0.5)
    b.recharge()
    assert b.level.tolist() == pytest.approx([0.87, 0.52, 1.0])


def test_n_transfers_by_mode():
    assert n_transfers(make_cfg()) == 3
    assert n_transfers(make_cfg(["fl.blend_mode=before_training"])) == 2


def _toy():
    n = np.array([100.0, 200.0, 300.0, 400.0])
    speed = np.array([1.0, 2.0, 4.0, 0.5])
    bw = np.array([1.0, 2.0, 4.0, 0.5])
    return n, speed, bw


def test_round_costs_formulas():
    n, speed, bw = _toy()
    comm, comp, lat = round_costs([1, 3], [2, 5], n, speed, bw, 1000.0, make_cfg())
    assert comm == pytest.approx(1500.0 + 6000.0)
    assert comp == pytest.approx(200.0 + 4000.0)
    assert lat == pytest.approx(6000.0 + 4000.0)
    comm2, _, _ = round_costs([1, 3], [2, 5], n, speed, bw, 1000.0,
                              make_cfg(["fl.blend_mode=before_training"]))
    assert comm2 == pytest.approx(1000.0 + 4000.0)
    assert round_costs([], [], n, speed, bw, 1000.0, make_cfg()) == (0.0, 0.0, 0.0)


def test_cost_reference():
    cfg = make_cfg()
    n, speed, bw = _toy()
    comm_ref, comp_ref = cost_reference(n, speed, bw, 1000.0, cfg)
    assert comm_ref == pytest.approx(cfg.fl.k * np.mean(1000.0 * 3 / bw))
    assert comp_ref == pytest.approx(cfg.fl.k * np.mean(2 * n / speed))


def test_random_choices_normalize_to_about_one():
    cfg = make_cfg()
    rng = np.random.default_rng(0)
    N = 20
    n = np.clip(rng.lognormal(np.log(600) - 0.125, 0.5, N), 150, 2000)
    speed, bw = rng.lognormal(0, 0.5, N), rng.lognormal(0, 0.5, N)
    mb = 46730 * 4.0
    refs = cost_reference(n, speed, bw, mb, cfg)
    vals = []
    for _ in range(4000):
        ch = rng.choice(N, cfg.fl.k, replace=False)
        comm, comp, _ = round_costs(ch, [2] * cfg.fl.k, n, speed, bw, mb, cfg)
        vals.append(normalize_costs(comm, comp, refs, cfg))
    mean = np.mean(vals, axis=0)
    assert mean == pytest.approx([1.0, 1.0], abs=0.05)


def test_normalize_clip():
    cfg = make_cfg()
    assert normalize_costs(10.0, -1.0, (1.0, 1.0), cfg) == (cfg.reward.cost_clip, 0.0)
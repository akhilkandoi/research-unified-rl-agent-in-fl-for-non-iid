"""Tests for fedrl/env/reward.py (mock RoundResult via SimpleNamespace)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from fedrl.config import load_config
from fedrl.env.reward import compute_reward, route_rewards

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(str(ROOT / "configs" / "default.yaml"), [])


def mock(**kw):
    base = dict(global_val_acc_before=0.5, global_val_acc_after=0.6, pers_gain_chosen=0.1,
                fair_p10_before=0.2, fair_p10_after=0.3, comm=2.0, comp=3.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_terms_and_total():
    rc = CFG.reward
    total, t = compute_reward(mock(), (4.0, 2.0), CFG, last_round=False)
    s = rc.acc_scale
    assert t["r_global"] == pytest.approx(s * 0.1)
    assert t["r_pers"] == pytest.approx(s * 0.1)
    assert t["r_fair"] == pytest.approx(s * 0.1)
    assert t["comm_norm"] == pytest.approx(0.5) and t["comp_norm"] == pytest.approx(1.5)
    assert t["final_bonus"] == 0.0
    expected = (rc.w_global * t["r_global"] + rc.w_pers * t["r_pers"] + rc.w_fair * t["r_fair"]
                - rc.w_comm * 0.5 - rc.w_comp * 1.5)
    assert total == pytest.approx(expected)


def test_final_bonus_only_last_round():
    _, t0 = compute_reward(mock(), (4.0, 2.0), CFG, last_round=False)
    total1, t1 = compute_reward(mock(), (4.0, 2.0), CFG, last_round=True)
    assert t0["final_bonus"] == 0.0
    assert t1["final_bonus"] == pytest.approx(CFG.reward.final_bonus * 0.6)
    assert total1 == pytest.approx(compute_reward(mock(), (4.0, 2.0), CFG, False)[0] + t1["final_bonus"])


def test_cost_clipping():
    _, t = compute_reward(mock(comm=1e6, comp=0.0), (1.0, 1.0), CFG, False)
    assert t["comm_norm"] == CFG.reward.cost_clip and t["comp_norm"] == 0.0


def test_alpha_zero_gives_exactly_zero_pers():
    _, t = compute_reward(mock(pers_gain_chosen=0.0), (1.0, 1.0), CFG, False)
    assert t["r_pers"] == 0.0


def test_nan_selection_is_safe():
    total, t = compute_reward(mock(pers_gain_chosen=float("nan")), (1.0, 1.0), CFG, False)
    assert t["r_pers"] == 0.0 and total == total


def test_routing():
    rc = CFG.reward
    total, t = compute_reward(mock(), (4.0, 2.0), CFG, last_round=True)
    assert route_rewards(t, CFG, "joint") == {"joint": pytest.approx(total)}
    shared = route_rewards(t, CFG, "stacked_shared")
    assert set(shared) == {"select", "train", "mix"}
    assert all(v == pytest.approx(total) for v in shared.values())
    sp = route_rewards(t, CFG, "stacked_specialized")
    bonus = rc.w_global * t["final_bonus"]
    assert sp["select"] == pytest.approx(rc.w_global * t["r_global"] + rc.w_fair * t["r_fair"] + bonus)
    assert sp["train"] == pytest.approx(rc.w_global * t["r_global"] - rc.w_comp * t["comp_norm"] + bonus)
    assert sp["mix"] == pytest.approx(rc.w_pers * t["r_pers"])
    with pytest.raises(ValueError):
        route_rewards(t, CFG, "nope")
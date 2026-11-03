from pathlib import Path

import numpy as np
import pytest
import torch

from fedrl.config import load_config
from fedrl.data.datasets import DatasetBundle
from fedrl.data.pools import build_pools
from fedrl.data.scenario import Scenario
from fedrl.data.scenario_sets import ScenarioSpec
from fedrl.fl.costs import n_transfers
from fedrl.fl.evaluate import evaluate_vec
from fedrl.fl.fl_round import FLWorld, IneligibleClientError, make_action
from fedrl.fl.resources import drain, eligible_mask
from fedrl.utils.flat import blend

ROOT = Path(__file__).resolve().parents[1]
DEV = torch.device("cpu")


def make_cfg(overrides=None):
    return load_config(str(ROOT / "configs" / "smoke.yaml"),
                       overrides=["run.device=cpu"] + list(overrides or []))


@pytest.fixture(scope="module")
def bundle():
    g = torch.Generator().manual_seed(0)
    protos = torch.randn(10, 1, 28, 28, generator=g)

    def mk(n):
        y = torch.randint(0, 10, (n,), generator=g)
        return protos[y] + 0.2 * torch.randn(n, 1, 28, 28, generator=g), y

    tx, ty = mk(6000)
    ex, ey = mk(1000)
    return DatasetBundle(name="mnist", num_classes=10, train_x=tx, train_y=ty,
                         test_x=ex, test_y=ey, device=DEV)


def make_world(bundle, cfg, a=1.0, drift="none", seed=3):
    pools = build_pools(bundle, cfg)
    scen = Scenario(ScenarioSpec(a=a, drift=drift, seed=seed), bundle, pools, cfg)
    return FLWorld(scen, bundle, pools, cfg, DEV)


def pers_acc(world, i, vec):
    idx = world.scenario.clients[i].personal_val_idx
    return evaluate_vec(world.model, vec, world.X, world.y, idx)[1]


def test_reset_and_probe(bundle):
    w = make_world(bundle, make_cfg())
    assert w.G.numel() == 46730
    assert all(l is None for l in w.L)
    loss, acc = w.probe_all()
    assert loss.shape == (w.N,) and acc.shape == (w.N,)
    assert (w.battery >= 0.5).all() and (w.battery <= 1.0).all()


def test_blend_after_aggregation_matches_manual(bundle):
    cfg = make_cfg()
    w = make_world(bundle, cfg)
    act = make_action(cfg, w.N, [0, 1], lr_idx=2, alpha_value=[0.3] * w.N)
    res = w.run_round(1, act)
    gains = [pers_acc(w, i, blend(0.3, w.G, w.L[i])) - pers_acc(w, i, w.L[i]) for i in [0, 1]]
    assert res.pers_gain_chosen == pytest.approx(float(np.mean(gains)), abs=1e-9)


def test_alpha_zero_gives_exactly_zero_gain(bundle):
    cfg = make_cfg()
    w = make_world(bundle, cfg)
    res = w.run_round(1, make_action(cfg, w.N, [0, 1], alpha_value=[0.0] * w.N))
    assert res.pers_gain_chosen == 0.0
    assert res.pers_gain_all == 0.0


def test_never_trained_client_uses_global(bundle):
    cfg = make_cfg()
    w = make_world(bundle, cfg)
    res = w.run_round(1, make_action(cfg, w.N, [0], lr_idx=2))
    assert w.L[5] is None
    assert torch.equal(w.personalized_vec(5, 0.3), w.G)
    assert res.pers_gain_all == pytest.approx(res.pers_gain_chosen / w.N, abs=1e-9)


def test_counters(bundle):
    cfg = make_cfg()
    w = make_world(bundle, cfg)
    w.run_round(1, make_action(cfg, w.N, [0, 1]))
    r = w.run_round(2, make_action(cfg, w.N, [1, 2]))
    assert r.times_picked[:4].tolist() == [1, 2, 1, 0]
    assert r.idle_rounds[:4].tolist() == [1, 0, 0, 2]
    assert r.rounds_since_local[:4].tolist() == [1, 0, 0, 2]


def test_battery_update_in_round(bundle):
    cfg = make_cfg()
    w = make_world(bundle, cfg)
    before = w.battery.clone()
    res = w.run_round(1, make_action(cfg, w.N, [0], epochs_idx=1))
    exp0 = min(float(before[0]) - float(drain(2, w.n[0], cfg)) + cfg.clients.recharge, 1.0)
    assert res.battery[0].item() == pytest.approx(exp0, abs=1e-6)
    assert res.battery[1].item() == pytest.approx(min(float(before[1]) + cfg.clients.recharge, 1.0), abs=1e-6)


@pytest.mark.parametrize("mode", ["after_aggregation", "before_training"])
def test_blend_modes_run_and_comm_uses_transfers(bundle, mode):
    cfg = make_cfg([f"fl.blend_mode={mode}"])
    w = make_world(bundle, cfg)
    for t in (1, 2, 3):
        res = w.run_round(t, make_action(cfg, w.N, [0, 1]))
        assert torch.isfinite(w.G).all()
        assert np.isfinite(res.pers_gain_chosen)
    exp = sum(w.model_bytes * n_transfers(cfg) / float(w.bw[i]) for i in [0, 1])
    assert res.comm == pytest.approx(exp, rel=1e-5)
    assert n_transfers(cfg) == (3 if mode == "after_aggregation" else 2)


def test_ineligible_choice_raises(bundle):
    cfg = make_cfg()
    w = make_world(bundle, cfg)
    w.bat.level[3] = 0.0
    with pytest.raises(IneligibleClientError):
        w.run_round(1, make_action(cfg, w.N, [3]))


def test_masked_epochs_rejected(bundle):
    cfg = make_cfg()
    w = make_world(bundle, cfg)
    w.bat.level[2] = float(cfg.clients.min_battery + drain(1, w.n[2], cfg) + 1e-3)
    assert eligible_mask(w.battery, w.n, cfg)[2]
    with pytest.raises(IneligibleClientError):
        w.run_round(1, make_action(cfg, w.N, [2], epochs_idx=2))
    w.run_round(1, make_action(cfg, w.N, [2], epochs_idx=0))


def test_duplicate_selection_raises(bundle):
    cfg = make_cfg()
    w = make_world(bundle, cfg)
    with pytest.raises(ValueError):
        w.run_round(1, make_action(cfg, w.N, [1, 1]))


def test_identical_seeds_identical_worlds(bundle):
    cfg = make_cfg()
    w1, w2 = make_world(bundle, cfg), make_world(bundle, cfg)
    assert torch.equal(w1.G, w2.G)
    act = make_action(cfg, w1.N, [0, 2], lr_idx=2)
    w1.run_round(1, act)
    w2.run_round(1, act)
    assert torch.equal(w1.G, w2.G)
    assert not torch.equal(make_world(bundle, cfg, seed=4).G, w1.G)


def test_apply_drift_if_due_reprobes(bundle):
    cfg = make_cfg()
    w = make_world(bundle, cfg, a=0.3, drift="abrupt")
    assert w.apply_drift_if_due(1) is False
    n0 = w.n_probes
    assert w.apply_drift_if_due(int(cfg.drift.abrupt_round)) is True
    assert w.n_probes == n0 + 1


def test_five_round_run_improves_global_val_accuracy(bundle):
    cfg = make_cfg()
    w = make_world(bundle, cfg, a=1.0)
    rng = np.random.default_rng(0)
    acc0 = w.global_val_acc()
    for t in range(1, 6):
        elig = np.flatnonzero(eligible_mask(w.battery, w.n, cfg).numpy())
        pick = [int(i) for i in rng.choice(elig, size=cfg.fl.k, replace=False)]
        res = w.run_round(t, make_action(cfg, w.N, pick, lr_idx=2, epochs_idx=2, batch_idx=0))
    assert res.global_val_acc_after > acc0 + 0.05
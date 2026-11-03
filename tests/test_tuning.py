"""Tuning budget, validation-only scenarios and best-baseline selection."""
import json

import pytest

from fedrl.config import load_config
from fedrl.controllers import build_controller
from fedrl.controllers.fixed import (NON_RL_BASELINES, BudgetExceeded, record_validation_score,
                                     select_best_baseline, tune_alpha, tune_settings)
from fedrl.data.scenario_sets import test_scenarios, validation_scenarios
from fedrl.env.state import state_dim



def _cfg(budget):
    return load_config("configs/smoke.yaml", [f"tuning.budget_rounds={budget}"])


class FakeEvaluator:
    """Scores candidates by closeness to a target; records every scenario it is given."""

    def __init__(self, cfg, score_fn):
        self.t, self.score_fn, self.specs, self.calls = int(cfg.fl.T), score_fn, [], 0

    def __call__(self, controller, spec):
        self.specs.append(spec)
        self.calls += 1
        return {"return": self.score_fn(controller), "rounds": self.t}


def _settings_score(c):
    return -(abs(c.lr_idx - 2) + abs(c.epochs_idx - 0) + abs(c.batch_idx - 1))


def test_tune_settings_within_budget_and_writes_json(tmp_path):
    cfg = _cfg(10_000)
    n_scen = len(validation_scenarios(cfg))
    ev = FakeEvaluator(cfg, _settings_score)
    res = tune_settings(cfg, evaluator=ev, run_dir=tmp_path)
    assert res["rounds_used"] == 27 * n_scen * cfg.fl.T <= res["budget_cap"]
    assert res["chosen"] == {"lr_idx": 2, "epochs_idx": 0, "batch_idx": 1}
    data = json.loads((tmp_path / "tuning.json").read_text())
    assert data["rounds_used"] == res["rounds_used"] and len(data["candidates"]) == 27
    ctrl = build_controller("tuned_settings", cfg, state_dim(cfg), run_dir=tmp_path)
    assert (ctrl.lr_idx, ctrl.epochs_idx, ctrl.batch_idx) == (2, 0, 1)


def test_tune_raises_when_cap_would_be_exceeded_before_running(tmp_path):
    cfg = _cfg(1)
    cfg_exact = load_config("configs/smoke.yaml")
    need = 27 * len(validation_scenarios(cfg_exact)) * cfg_exact.fl.T
    tight = _cfg(need - 1)
    ev = FakeEvaluator(tight, _settings_score)
    with pytest.raises(BudgetExceeded):
        tune_settings(tight, evaluator=ev, run_dir=tmp_path)
    assert ev.calls == 0
    assert not (tmp_path / "tuning.json").exists()
    with pytest.raises(BudgetExceeded):
        tune_alpha(cfg, evaluator=FakeEvaluator(cfg, lambda c: 0.0), run_dir=tmp_path)
    ok = _cfg(need)
    tune_settings(ok, evaluator=FakeEvaluator(ok, _settings_score), run_dir=tmp_path)


def test_budget_charged_by_reported_rounds(tmp_path):
    cfg = _cfg(5 * len(validation_scenarios(load_config("configs/smoke.yaml"))) * 5)

    def greedy_eval(controller, spec):
        return {"return": 0.0, "rounds": 10 * int(cfg.fl.T)}  # reports more than planned

    with pytest.raises(BudgetExceeded):
        tune_alpha(cfg, evaluator=greedy_eval, run_dir=tmp_path)


def test_tuning_reads_only_validation_scenarios(tmp_path):
    cfg = _cfg(10_000)
    ev = FakeEvaluator(cfg, lambda c: -abs(c.alpha_idx - 3))
    res = tune_alpha(cfg, evaluator=ev, run_dir=tmp_path)
    val, test = set(validation_scenarios(cfg)), set(test_scenarios(cfg))
    assert ev.specs and set(ev.specs) <= val and not (set(ev.specs) & test)
    assert res["chosen"] == {"alpha_idx": 3}
    assert res["rounds_used"] == 5 * len(validation_scenarios(cfg)) * cfg.fl.T
    with pytest.raises(ValueError):
        tune_alpha(cfg, evaluator=ev, scenarios=list(test_scenarios(cfg))[:1], run_dir=tmp_path)


def test_select_best_baseline_reads_validation_scores(tmp_path):
    cfg = _cfg(10_000)
    path = tmp_path / "validation_scores.json"
    with pytest.raises(FileNotFoundError):
        select_best_baseline(cfg, path=path)
    for i, name in enumerate(NON_RL_BASELINES):
        record_validation_score(cfg, name, float(i), {}, path=path)
    assert select_best_baseline(cfg, path=path) == NON_RL_BASELINES[-1]
    record_validation_score(cfg, "fedprox", 99.0, {}, path=path)
    assert select_best_baseline(cfg, path=path) == "fedprox"
    partial = tmp_path / "partial.json"
    record_validation_score(cfg, "fedavg", 1.0, {}, path=partial)
    with pytest.raises(ValueError):
        select_best_baseline(cfg, path=partial)
    assert select_best_baseline(cfg, path=partial, allow_partial=True) == "fedavg"
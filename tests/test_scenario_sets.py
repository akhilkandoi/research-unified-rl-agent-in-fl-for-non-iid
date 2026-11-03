import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest
from helpers_data import make_cfg

from fedrl.data.scenario_sets import (ScenarioSpec, assert_disjoint, check_disjoint, test_conditions,
                                      test_scenarios, training_scenario, validation_scenarios)


def test_sets_disjoint_and_sized():
    cfg = make_cfg()
    assert_disjoint(cfg)
    tr = {training_scenario(e, cfg).seed for e in range(300)}
    va = {s.seed for s in validation_scenarios(cfg)}
    te = {s.seed for s in test_scenarios(cfg)}
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(validation_scenarios(cfg)) == 12 and len(test_scenarios(cfg)) == 45
    assert len(test_conditions(cfg)) == 9
    assert {s.a for s in test_scenarios(cfg)}.isdisjoint({0.1, 0.3, 1.0})
    assert {s.drift for s in test_scenarios(cfg)} == {"none", "abrupt", "gradual"}


def test_training_scenario_deterministic_and_in_sets():
    cfg = make_cfg()
    assert training_scenario(5, cfg) == training_scenario(5, cfg)
    assert training_scenario(5, cfg).seed == 5 and training_scenario(105, cfg).seed == 5
    combos = {(s.a, s.drift) for s in (training_scenario(e, cfg) for e in range(600))}
    assert combos == {(a, d) for a in (0.1, 0.3, 1.0) for d in ("none", "abrupt")}
    assert all(0 <= training_scenario(e, cfg).seed < 100 for e in range(600))
    assert isinstance(training_scenario(0, cfg), ScenarioSpec)


def test_check_disjoint_detects_problems():
    check_disjoint(range(100), [500], [1000], [0.1], [0.2])
    with pytest.raises(ValueError):
        check_disjoint(range(100), [99], [1000], [0.1], [0.2])
    with pytest.raises(ValueError):
        check_disjoint(range(100), [500], [500], [0.1], [0.2])
    with pytest.raises(ValueError):
        check_disjoint(range(100), [500], [1000], [0.1, 0.2], [0.2])
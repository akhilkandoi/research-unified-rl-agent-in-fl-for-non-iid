"""Disjoint train / validation / test scenario sets (MASTER 2.11)."""
from __future__ import annotations

from dataclasses import dataclass

from fedrl.data.drift import DRIFT_KINDS
from fedrl.utils.seeding import make_rng


@dataclass(frozen=True)
class ScenarioSpec:
    """One scenario: Dirichlet concentration ``a``, drift kind and scenario seed."""

    a: float
    drift: str
    seed: int


def _grid(a_list, drifts, seeds) -> list[ScenarioSpec]:
    return [ScenarioSpec(float(a), str(d), int(s)) for a in a_list for d in drifts for s in seeds]


def training_scenario(episode: int, cfg) -> ScenarioSpec:
    """Scenario for RL episode ``episode``: seed = episode mod n_seeds; (a, drift) drawn from the
    training sets by make_rng(seed, 'train_scenario_<episode>'), hence deterministic."""
    tr = cfg.scenarios.train
    seed = int(episode) % int(tr.n_seeds)
    rng = make_rng(seed, f"train_scenario_{int(episode)}")
    a_list, d_list = list(tr.a), list(tr.drift)
    a = a_list[int(rng.integers(len(a_list)))]
    d = d_list[int(rng.integers(len(d_list)))]
    return ScenarioSpec(float(a), str(d), seed)


def validation_scenarios(cfg) -> list[ScenarioSpec]:
    """All validation scenarios (a x drift x seeds)."""
    v = cfg.scenarios.val
    return _grid(v.a, v.drift, v.seeds)


def test_scenarios(cfg) -> list[ScenarioSpec]:
    """All test scenarios (a x drift x evaluation seeds). Never used for training or tuning."""
    t = cfg.scenarios.test
    return _grid(t.a, t.drift, t.seeds)


test_scenarios.__test__ = False  # not a pytest test when imported into test modules


def test_conditions(cfg) -> list[tuple[float, str]]:
    """The (a, drift) conditions of the test set (9 in the default config)."""
    t = cfg.scenarios.test
    return [(float(a), str(d)) for a in t.a for d in t.drift]


test_conditions.__test__ = False


def check_disjoint(train_seeds, val_seeds, test_seeds, train_a, test_a) -> None:
    """Raise ValueError if seed sets overlap or a test skew value also appears in train."""
    tr, va, te = set(map(int, train_seeds)), set(map(int, val_seeds)), set(map(int, test_seeds))
    for na, a, nb, b in (("train", tr, "val", va), ("train", tr, "test", te), ("val", va, "test", te)):
        if a & b:
            raise ValueError(f"Scenario seeds of '{na}' and '{nb}' overlap: {sorted(a & b)}")
    both = set(map(float, train_a)) & set(map(float, test_a))
    if both:
        raise ValueError(f"Test skew values also appear in train: {sorted(both)}")


def assert_disjoint(cfg) -> None:
    """Validate the configured scenario sets (seeds disjoint, test a not in train, valid drift names)."""
    s = cfg.scenarios
    for group in (s.train.drift, s.val.drift, s.test.drift):
        for d in group:
            if d not in DRIFT_KINDS:
                raise ValueError(f"Unknown drift '{d}' in scenarios config")
    check_disjoint(range(int(s.train.n_seeds)), s.val.seeds, s.test.seeds, s.train.a, s.test.a)
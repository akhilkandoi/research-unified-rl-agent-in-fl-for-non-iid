import numpy as np

from fedrl.eval.pilot import (criterion_a, criterion_b, criterion_c, evaluate_pilot_criteria, format_criteria,
                              moving_average, plateau_index, sweep_plateau)


def _rising(seeds=3, points=15):
    base = np.linspace(0, 10, points)
    return np.stack([base + 0.1 * s for s in range(seeds)])


def test_moving_average_trailing():
    assert np.allclose(moving_average([1, 2, 3, 4, 5, 6], 5), [1, 1.5, 2, 2.5, 3, 4])


def test_plateau_detects_flat_tail():
    c = np.concatenate([np.linspace(0, 10, 6), np.full(9, 10.0)])
    c = np.stack([c, c + 0.2, c - 0.2])
    p, _ = plateau_index(c)
    assert p <= 9
    res = sweep_plateau({2: c, 4: _rising()}, min_tail=0.2)
    assert res["per_u"][2]["plateaus"] and not res["per_u"][4]["plateaus"] and res["selected"] == 2


def test_criterion_a_pass_and_fail():
    assert criterion_a(_rising())["pass"]
    flat = np.stack([np.ones(15) + 0.5 * s for s in range(3)])
    flat[0, -3:] += 0.1
    assert not criterion_a(np.stack([np.full(15, 1.0), np.full(15, 1.0) + 0.3, np.full(15, 1.0) - 0.3]))["pass"]
    assert not criterion_a(-_rising())["pass"]


def test_criterion_b():
    ok = [0.5 + 0.1 * np.sin(np.arange(50) / 5) for _ in range(3)]
    assert criterion_b(ok)["pass"]
    assert not criterion_b([np.full(50, 0.5)] * 3)["pass"]                 # constant
    assert not criterion_b([np.linspace(0.1, 0.95, 50)] * 3)["pass"]       # leaves [0.2, 0.8]
    assert not criterion_b([])["pass"]


def test_criterion_c():
    uniform = [[6] * 10 for _ in range(4)]
    assert criterion_c(uniform)["pass"]
    starved = [[30, 30, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(4)]
    assert not criterion_c(starved)["pass"]
    hogging = [[20] + [4] * 9 for _ in range(4)]       # one client 3.4x uniform share
    assert not criterion_c(hogging)["pass"]


def test_overall_and_remedies_text():
    good = evaluate_pilot_criteria(_rising(), [0.5 + 0.1 * np.sin(np.arange(50) / 5)] * 3, [[6] * 10] * 4)
    assert good["all_pass"] and "OVERALL: PASS" in format_criteria(good)
    bad = evaluate_pilot_criteria(-_rising(), [np.full(50, 0.5)] * 3, [[30, 30] + [0] * 8] * 4)
    txt = format_criteria(bad)
    assert not bad["all_pass"] and "FAIL" in txt and txt.index("per-client credit assignment") < txt.index("different memory")
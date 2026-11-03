import numpy as np
import pandas as pd
import pytest

from fedrl.eval.stats import compare_methods, paired_bootstrap, select_best_baseline, summarize


def test_bootstrap_clear_effect_excludes_zero_and_covers_truth():
    rng = np.random.default_rng(0)
    b = rng.normal(0.5, 0.1, 60)
    a = b + 0.5 + rng.normal(0, 0.05, 60)
    r = paired_bootstrap(a, b, 10000, 0.95, np.random.default_rng(1))
    assert r["excludes_zero"] and r["lo"] <= 0.5 <= r["hi"]


def test_bootstrap_no_effect_includes_zero():
    x = np.random.default_rng(0).normal(0, 0.1, 100)
    a = np.concatenate([x, -x]) + 0.7
    b = np.full_like(a, 0.7)
    r = paired_bootstrap(a, b, 10000, 0.95, np.random.default_rng(2))
    assert not r["excludes_zero"] and r["lo"] < 0 < r["hi"]


def test_bootstrap_requires_matched_arrays():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(3), np.zeros(4), 100, 0.95, np.random.default_rng(0))


def _synthetic(shuffle=False):
    rng = np.random.default_rng(3)
    base = {(d, e): rng.normal(0.8, 0.1) for d in ("none", "abrupt") for e in range(1000, 1005)}
    rows = []
    for method, off in (("joint", 0.05), ("stacked_shared", 0.0), ("fedavg", 0.0)):
        for ts in (0, 1):
            for (d, e), v in base.items():
                rows.append({"dataset": "mnist", "method": method, "variant": "default", "train_seed": ts,
                             "a": 0.2, "drift": d, "eval_seed": e, "pers_test_acc": v + off})
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=0) if shuffle else df


def test_matched_pairing_gives_tight_ci_and_is_order_independent():
    res = compare_methods(_synthetic(), best_baseline="fedavg", n_resamples=500, print_count=False)
    assert res["excludes_zero"].all() and (res["lo"] > 0.049).all()
    res2 = compare_methods(_synthetic(shuffle=True), best_baseline="fedavg", n_resamples=500, print_count=False)
    assert np.allclose(res["mean_diff"].to_numpy(), res2["mean_diff"].to_numpy())


def test_compare_counts_and_verdicts():
    res = compare_methods(_synthetic(), best_baseline="fedavg", n_resamples=200, print_count=False)
    # (stacked_shared + best baseline + fedavg) x (2 scenarios + pooled)
    assert len(res) == 9 and res.attrs["n_comparisons"] == 9
    assert set(res["verdict"]) == {"better"}


def test_tie_when_no_difference():
    df = _synthetic()
    df.loc[df["method"] == "joint", "pers_test_acc"] -= 0.05
    res = compare_methods(df, best_baseline="fedavg", n_resamples=200, print_count=False)
    assert set(res["verdict"]) == {"tie"}


def test_summarize_mean_std_and_per_seed():
    df = _synthetic()
    s = summarize(df, ["pers_test_acc"], by=("dataset", "method", "variant", "a", "drift"))
    row = s[(s.method == "joint") & (s.drift == "none")].iloc[0]
    assert row["n_seeds"] == 2 and set(row["pers_test_acc_per_seed"]) == {0, 1}


def test_select_best_baseline_uses_validation_scores_only():
    assert select_best_baseline({"fedavg": 1.0, "loss_heuristic": 2.0, "joint": 99.0}) == "loss_heuristic"
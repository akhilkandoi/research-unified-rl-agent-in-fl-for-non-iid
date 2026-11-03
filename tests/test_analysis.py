import json

import numpy as np
import pandas as pd

from fedrl.eval.analysis import (load_result_records, methods_paragraph, results_dataframe, run_analysis,
                                 scenario_statements, LOCAL_ONLY_SENTENCE)


def _write_result(root, method, ts, a, drift, es, acc, comm=10.0):
    d = root / "mnist" / method / "default" / f"seed{ts}" / "abc123" / "results"
    d.mkdir(parents=True, exist_ok=True)
    res = {"metrics": {"global_test_acc": acc, "pers_test_acc": acc + 0.01, "worst10_acc": acc - 0.1,
                       "min_acc": acc - 0.2, "comm": comm, "comp": 20.0, "latency": 3.0,
                       "rounds_to_target": 5, "auc": acc, "drift_recovery": None if drift == "none" else 2,
                       "drift_start": None if drift == "none" else 15},
           "diagnostics": {"pers_gain_chosen_per_round": [0.01] * 5, "pers_gain_all_per_round": [0.005] * 5,
                           "mean_alpha_per_round": [0.5] * 5, "selection_counts": [3, 2, 1, 0]},
           "curves": {"global_test_acc": list(np.linspace(0.3, acc, 30))},
           "meta": {"dataset": "mnist", "method": method, "variant": "default", "train_seed": ts,
                    "scenario": {"a": a, "drift": drift, "seed": es}, "eval_seed": es, "tag": "",
                    "N": 20, "k": 5, "T": 30}}
    (d / f"a{a:g}_{drift}_s{es}_e{es}.json").write_text(json.dumps(res))


def _build(root):
    rng = np.random.default_rng(0)
    for drift in ("none", "abrupt"):
        for es in (1000, 1001, 1002):
            base = 0.8 + rng.normal(0, 0.02)
            for method, off, seeds in (("joint", 0.05, (0, 1)), ("stacked_shared", 0.0, (0, 1)),
                                       ("stacked_specialized", 0.0, (0, 1)), ("fedavg", 0.0, (0,)),
                                       ("fixed_alpha_0.5", 0.0, (0,))):
                for ts in seeds:
                    _write_result(root, method, ts, 0.2, drift, es, base + off)
    for m, score in (("fedavg", 1.0), ("fixed_alpha_0.5", 2.0)):
        d = root / "mnist" / m / "default" / "seed0" / "abc123"
        d.mkdir(parents=True, exist_ok=True)
        (d / "validation.json").write_text(json.dumps({"mean_return": score, "method": m, "dataset": "mnist"}))


def test_loading(tmp_path):
    _build(tmp_path)
    recs = load_result_records(tmp_path)
    assert len(recs) == 2 * 3 * (2 + 2 + 2 + 1 + 1)
    df = results_dataframe(tmp_path)
    assert {"pers_gain_chosen_mean", "method", "eval_seed"} <= set(df.columns)


def test_analysis_runs_on_tiny_results(tmp_path):
    _build(tmp_path)
    out = tmp_path / "out"
    res = run_analysis(tmp_path, out, cfg=None, n_resamples=200, include_param_table=False)
    assert res["n_comparisons"] > 0
    for name in ("summary_mnist.md", "per_seed_mnist.csv", "compare_mnist_pers_test_acc.csv",
                 "statements.txt", "methods.md", "best_baseline_mnist.json"):
        assert (out / name).exists(), name
    assert json.loads((out / "best_baseline_mnist.json").read_text())["best"] == "fixed_alpha_0.5"
    text = (out / "methods.md").read_text()
    assert LOCAL_ONLY_SENTENCE in text and f"{res['n_comparisons']} comparisons" in text
    assert "BETTER than" in (out / "statements.txt").read_text()
    assert any(p.suffix == ".png" for p in out.iterdir())


def test_statements_never_hide_ties_or_losses():
    cmp = pd.DataFrame([
        {"metric": "m", "a": 0.2, "drift": "none", "other": "x", "verdict": "better"},
        {"metric": "m", "a": 0.2, "drift": "none", "other": "y", "verdict": "tie"},
        {"metric": "m", "a": 0.2, "drift": "none", "other": "z", "verdict": "worse"}])
    s = scenario_statements(cmp)[0]
    assert "BETTER than: x" in s and "TIES with: y" in s and "WORSE than: z" in s


def test_methods_paragraph_mentions_local_only_and_count():
    df = pd.DataFrame({"method": ["joint"], "train_seed": [0], "eval_seed": [1000]})
    t = methods_paragraph(42, df, 10000, 0.95)
    assert "42 comparisons" in t and "not a persistent personal model" in t
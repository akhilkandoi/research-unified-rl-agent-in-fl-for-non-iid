import numpy as np
import torch

from fedrl.config import load_config
from fedrl.eval import metrics as M


def test_rounds_to_target():
    assert M.rounds_to_target([0.1, 0.5, 0.9], 0.5) == 2
    assert M.rounds_to_target([0.1, 0.5, 0.9], 0.95) is None
    assert M.rounds_to_target([0.1], None) is None


def test_auc_is_mean():
    assert abs(M.area_under_curve([0.0, 0.5, 1.0]) - 0.5) < 1e-12


def test_drift_recovery_abrupt():
    curve = [0.5, 0.6, 0.7, 0.8, 0.4, 0.5, 0.7, 0.8, 0.85]
    assert M.drift_recovery_time(curve, 5, 5) == 3  # regains 0.8 at round 8


def test_drift_recovery_never_and_none():
    assert M.drift_recovery_time([0.5, 0.8, 0.4, 0.5], 3, 3) is None
    assert M.drift_recovery_time([0.1, 0.5, 0.6, 0.7], 3, 3) == 0


def test_drift_recovery_gradual_waits_for_end():
    curve = [0.5, 0.6, 0.5, 0.4, 0.62, 0.7]
    assert M.drift_recovery_time(curve, 3, 5) == 2


def test_drift_window():
    cfg = load_config(None)
    assert M.drift_window(cfg, "none") is None
    assert M.drift_window(cfg, "abrupt") == (15, 15)
    assert M.drift_window(cfg, "gradual") == (15, 22)


def test_client_stats_and_selection():
    s = M.client_accuracy_stats(list(np.linspace(0, 1, 11)), 10)
    assert s["min"] == 0.0 and abs(s["worst"] - 0.1) < 1e-9
    sel = M.selection_summary([5, 5, 0, 0])
    assert sel["coverage"] == 0.5 and abs(sel["max_ratio"] - 2.0) < 1e-9


def test_to_jsonable_handles_nan_and_tensors():
    out = M.to_jsonable({"a": torch.tensor([1.0, float("nan")]), "b": np.float32(2.0)})
    assert out == {"a": [1.0, None], "b": 2.0}
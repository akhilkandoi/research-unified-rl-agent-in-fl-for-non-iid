import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fedrl.fl.evaluate import evaluate_many, evaluate_vec, per_class_acc
from fedrl.models.cnn import build_model
from fedrl.utils.flat import get_flat, set_flat


def _tiny():
    model = nn.Linear(2, 2, bias=False)
    vec = torch.tensor([1.0, 0.0, 0.0, 1.0])  # identity weight -> logits == x
    X = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 0.0]])
    y = torch.tensor([0, 0, 1])
    return model, vec, X, y


def test_evaluate_vec_tiny_manual():
    model, vec, X, y = _tiny()
    loss, acc = evaluate_vec(model, vec, X, y, torch.arange(3), batch=2)
    manual = (math.log(math.exp(2) + 1) - 2 + math.log(1 + math.exp(3)) + math.log(math.e + 1)) / 3
    assert loss == pytest.approx(manual, rel=1e-5)
    assert acc == pytest.approx(1 / 3)


def test_evaluate_many_and_subset():
    model, vec, X, y = _tiny()
    out = evaluate_many(model, vec, X, y, [torch.tensor([0]), torch.tensor([1, 2])])
    assert out[0][1] == 1.0 and out[1][1] == 0.0
    assert out[0][0] == pytest.approx(math.log(math.exp(2) + 1) - 2, rel=1e-5)


def test_per_class_acc_tiny():
    model, vec, X, y = _tiny()
    acc = per_class_acc(model, vec, X, y, torch.arange(3), num_classes=3, batch=2)
    assert acc[0] == pytest.approx(0.5) and acc[1] == 0.0 and math.isnan(acc[2].item())


def test_cnn_matches_manual_and_restores_mode():
    torch.manual_seed(0)
    model = build_model("mnist")
    vec = torch.randn_like(get_flat(model)) * 0.05
    X = torch.randn(37, 1, 28, 28)
    y = torch.randint(0, 10, (37,))
    idx = torch.arange(5, 37)
    model.train()
    loss, acc = evaluate_vec(model, vec, X, y, idx, batch=7)
    assert model.training  # mode restored
    ref = build_model("mnist")
    set_flat(ref, vec)
    with torch.no_grad():
        logits = ref(X[idx])
        ref_loss = F.cross_entropy(logits, y[idx]).item()
        ref_acc = (logits.argmax(1) == y[idx]).float().mean().item()
    assert loss == pytest.approx(ref_loss, rel=1e-4)
    assert acc == pytest.approx(ref_acc)
    pc = per_class_acc(model, vec, X, y, idx, 10, batch=7)
    counts = torch.bincount(y[idx], minlength=10).float()
    valid = ~torch.isnan(pc)
    assert float((pc[valid] * counts[valid]).sum() / counts.sum()) == pytest.approx(ref_acc, abs=1e-6)


def test_empty_index_raises():
    model, vec, X, y = _tiny()
    with pytest.raises(ValueError, match="empty"):
        evaluate_vec(model, vec, X, y, torch.tensor([], dtype=torch.long))
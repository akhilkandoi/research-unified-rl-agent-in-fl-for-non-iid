"""Batched, no-grad model evaluation on flat parameter vectors (device-aware, no data copies)."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fedrl.utils.flat import set_flat


@contextmanager
def _eval_mode(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def _score(model: nn.Module, X: Tensor, y: Tensor, idx: Tensor, batch: int) -> tuple[float, float]:
    n = int(idx.numel())
    if n == 0:
        raise ValueError("Cannot evaluate on an empty index set.")
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}.")
    idx = idx.to(X.device)
    loss_sum = torch.zeros((), device=X.device)
    correct = torch.zeros((), device=X.device)
    for s in range(0, n, batch):
        ii = idx[s:s + batch]
        yb = y[ii]
        logits = model(X[ii])
        loss_sum += F.cross_entropy(logits, yb, reduction="sum")
        correct += (logits.argmax(dim=1) == yb).sum()
    return loss_sum.item() / n, correct.item() / n


def evaluate_vec(model: nn.Module, vec: Tensor, X: Tensor, y: Tensor, idx: Tensor,
                 batch: int = 1024) -> tuple[float, float]:
    """Load `vec` into `model` and return (mean cross-entropy loss, accuracy) on X[idx], y[idx]."""
    model.to(X.device)
    set_flat(model, vec)
    with _eval_mode(model):
        return _score(model, X, y, idx, batch)


def evaluate_many(model: nn.Module, vec: Tensor, X: Tensor, y: Tensor, idx_list: list[Tensor],
                  batch: int = 1024) -> list[tuple[float, float]]:
    """Evaluate one vector on several index sets (vector loaded once). Returns [(loss, acc), ...]."""
    model.to(X.device)
    set_flat(model, vec)
    with _eval_mode(model):
        return [_score(model, X, y, idx, batch) for idx in idx_list]


def per_class_acc(model: nn.Module, vec: Tensor, X: Tensor, y: Tensor, idx: Tensor,
                  num_classes: int, batch: int = 1024) -> Tensor:
    """Per-class accuracy Tensor[num_classes] on CPU; NaN for classes absent from idx."""
    n = int(idx.numel())
    if n == 0:
        raise ValueError("Cannot evaluate on an empty index set.")
    model.to(X.device)
    set_flat(model, vec)
    idx = idx.to(X.device)
    hits = torch.zeros(num_classes, device=X.device)
    tot = torch.zeros(num_classes, device=X.device)
    with _eval_mode(model):
        for s in range(0, n, batch):
            ii = idx[s:s + batch]
            yb = y[ii]
            pred = model(X[ii]).argmax(dim=1)
            tot += torch.bincount(yb, minlength=num_classes).float()
            hits += torch.bincount(yb[pred == yb], minlength=num_classes).float()
    acc = hits / tot.clamp_min(1.0)
    acc[tot == 0] = float("nan")
    return acc.cpu()
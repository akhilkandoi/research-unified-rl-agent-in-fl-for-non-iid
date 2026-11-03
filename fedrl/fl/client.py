"""Local client training: plain SGD, optional FedProx term."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from fedrl.utils.flat import get_flat, set_flat


def _split_like(vec: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    """Views of a flat vector with the shapes of `params` (FedProx anchor)."""
    out, start = [], 0
    for p in params:
        num = p.numel()
        out.append(vec[start:start + num].view_as(p).detach())
        start += num
    if start != vec.numel():
        raise ValueError(f"Vector has {vec.numel()} entries, model has {start}")
    return out


def local_train(model: torch.nn.Module, global_vec: torch.Tensor, X: torch.Tensor, y: torch.Tensor,
                idx: torch.Tensor, lr: float, epochs: int, batch: int, prox_mu: float,
                gen: torch.Generator, init_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Train `model` on X[idx], y[idx] with plain SGD and return the flat trained vector.

    Starts from `global_vec` (or `init_vec` if given, used by blend_mode=before_training).
    Per-epoch shuffling uses `gen`; the final partial batch is kept. The FedProx term
    (mu/2)*||w - global_vec||^2 is added through its gradient mu*(w - global_vec); with
    prox_mu == 0 nothing is added, so the result equals plain SGD exactly.
    """
    if epochs < 1 or batch < 1 or lr <= 0:
        raise ValueError(f"Invalid training settings lr={lr}, epochs={epochs}, batch={batch}")
    idx = idx.to(X.device)
    m = int(idx.numel())
    if m == 0:
        raise ValueError("Client has no training samples")
    set_flat(model, global_vec if init_vec is None else init_vec)
    params = list(model.parameters())
    anchors = _split_like(global_vec, params) if prox_mu > 0 else None
    opt = torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0)
    Xc, yc = X[idx], y[idx]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(m, generator=gen, device=gen.device).to(Xc.device)
        for s in range(0, m, batch):
            b = perm[s:s + batch]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(Xc[b]), yc[b])
            loss.backward()
            if anchors is not None:
                with torch.no_grad():
                    for p, a in zip(params, anchors):
                        p.grad.add_(p.detach() - a, alpha=prox_mu)
            opt.step()
    return get_flat(model).detach().clone()
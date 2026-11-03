"""Flat 1-D parameter vector helpers."""
from __future__ import annotations

import torch
from torch import Tensor, nn


def num_params(model: nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def get_flat(model: nn.Module) -> Tensor:
    """Copy of all parameters as one 1-D tensor."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def set_flat(model: nn.Module, vec: Tensor) -> None:
    """Load a flat vector into the model's parameters (in place, no autograd)."""
    n = num_params(model)
    if vec.dim() != 1 or vec.numel() != n:
        raise ValueError(f"set_flat expected a 1-D vector of length {n}, got shape {tuple(vec.shape)}.")
    off = 0
    with torch.no_grad():
        for p in model.parameters():
            m = p.numel()
            p.copy_(vec[off:off + m].view_as(p))
            off += m


def weighted_average(vecs: Tensor, weights: Tensor) -> Tensor:
    """Weighted mean of rows of vecs[m,P] with weights[m] (normalized internally)."""
    if vecs.dim() != 2 or weights.dim() != 1 or vecs.shape[0] != weights.shape[0]:
        raise ValueError(f"Shape mismatch: vecs {tuple(vecs.shape)}, weights {tuple(weights.shape)}.")
    w = weights.to(device=vecs.device, dtype=vecs.dtype)
    s = w.sum()
    if not bool(s > 0):
        raise ValueError("weighted_average requires weights with a positive sum.")
    return (w / s) @ vecs


def blend(alpha: float | Tensor, g: Tensor, l: Tensor) -> Tensor:
    """alpha*g + (1-alpha)*l. A tensor alpha of shape [m] broadcasts over rows of [m,P] inputs."""
    if isinstance(alpha, Tensor) and alpha.dim() == 1 and g.dim() == 2:
        alpha = alpha.unsqueeze(-1)
    return alpha * g + (1.0 - alpha) * l


def rel_div(w: Tensor, g: Tensor) -> float:
    """Relative divergence ||w-g|| / ||g||."""
    return float((w - g).norm() / g.norm().clamp_min(1e-12))
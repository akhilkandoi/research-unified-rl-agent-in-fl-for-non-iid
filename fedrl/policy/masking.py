"""Masking and NaN-safe categorical utilities.

All functions work on the last dimension. Masked options get -inf *before*
sampling. Rows that have no valid option never produce NaN.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

NEG_INF = float("-inf")
_TINY = 1e-30


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits of masked-out (mask == False) entries to -inf."""
    return logits.masked_fill(~mask, NEG_INF)


def sanitize_option_mask(mask: torch.Tensor) -> torch.Tensor:
    """Rows with no valid option get option 0 enabled (used for clients that
    can never be chosen, so that sampling stays well defined)."""
    none_valid = ~mask.any(dim=-1, keepdim=True)
    first = torch.zeros_like(mask)
    first[..., 0] = True
    return mask | (none_valid & first)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log-softmax over masked-in entries; masked-out entries are -inf.

    Safe when a row has a single valid entry (gives 0) or none (all -inf, no NaN).
    """
    x = mask_logits(logits, mask)
    m = x.amax(dim=dim, keepdim=True).detach()
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    z = x - m
    s = torch.exp(z).sum(dim=dim, keepdim=True)
    lse = torch.log(s.clamp_min(_TINY))
    return torch.where(mask, z - lse, torch.full_like(z, NEG_INF))


def masked_entropy(logp: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Entropy of a distribution given by masked log-probs (0 for empty rows)."""
    safe = torch.where(mask, logp, torch.zeros_like(logp))
    p = torch.where(mask, safe.exp(), torch.zeros_like(safe))
    return -(p * safe).sum(dim=dim)


def _uniform(like: torch.Tensor, gen: Optional[torch.Generator]) -> torch.Tensor:
    if gen is None:
        return torch.rand(like.shape, device=like.device, dtype=like.dtype)
    u = torch.rand(like.shape, generator=gen, device=gen.device, dtype=like.dtype)
    return u.to(like.device)


def gumbel_like(like: torch.Tensor, gen: Optional[torch.Generator]) -> torch.Tensor:
    """Standard Gumbel noise with the shape/device/dtype of ``like``."""
    u = _uniform(like, gen).clamp(1e-10, 1.0 - 1e-7)
    return -torch.log(-torch.log(u))


def sample_categorical(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    gen: Optional[torch.Generator],
    deterministic: bool = False,
) -> torch.Tensor:
    """Sample (Gumbel-max) or arg-max among masked-in options; returns idx[...]."""
    if mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
    scores = logits.detach()
    if not deterministic:
        scores = scores + gumbel_like(scores, gen)
    return scores.masked_fill(~mask, NEG_INF).argmax(dim=-1)


def categorical_logp_entropy(
    logits: torch.Tensor, idx: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """log p(idx) and entropy under the masked categorical (shapes logits[...,M], idx[...])."""
    if mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
    logp_all = masked_log_softmax(logits, mask)
    i = idx.unsqueeze(-1)
    lp = logp_all.gather(-1, i).squeeze(-1)
    ok = mask.gather(-1, i).squeeze(-1)
    lp = torch.where(ok, lp, torch.zeros_like(lp))
    return lp, masked_entropy(logp_all, mask)
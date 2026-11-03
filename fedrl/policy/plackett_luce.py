"""Batched Plackett-Luce sampling of k clients without replacement.

log_prob = sum over picks of [logit_chosen - logsumexp(logits of not-yet-chosen
eligible clients)].  Entropy (D8) = sum of the per-step entropies along the
sampled path (an approximation of the entropy of the whole ordering).
Padded order entries are -1 and contribute nothing.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from fedrl.policy.masking import NEG_INF, gumbel_like, masked_entropy, masked_log_softmax


def order_to_mask(order: torch.Tensor, n: int) -> torch.Tensor:
    """order[B,k] (padded -1) -> bool[B,n] of chosen clients."""
    valid = (order >= 0).long()
    z = torch.zeros(order.shape[0], n, dtype=torch.long, device=order.device)
    return z.scatter_add(1, order.clamp_min(0), valid) > 0


def log_prob_of_order(
    logits: torch.Tensor,
    order: torch.Tensor,
    mask: torch.Tensor,
    return_entropy: bool = False,
):
    """Log-probability (and optionally path entropy) of stored orders.

    Args:
        logits: [B,N] selection logits.
        order: [B,k] long, picks in order, -1 padded.
        mask: [B,N] bool eligibility.
    Returns:
        logp[B] or (logp[B], entropy[B]).
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B,N], got {tuple(logits.shape)}")
    B, N = logits.shape
    remaining = mask.clone()
    logp = logits.new_zeros(B)
    ent = logits.new_zeros(B)
    for j in range(order.shape[1]):
        pick = order[:, j]
        safe = pick.clamp_min(0)
        lp_all = masked_log_softmax(logits, remaining)
        lp = lp_all.gather(1, safe[:, None]).squeeze(1)
        is_pick = pick >= 0
        ok = is_pick & remaining.gather(1, safe[:, None]).squeeze(1)
        zero = torch.zeros_like(lp)
        impossible = is_pick & ~ok  # real pick of an ineligible or already-chosen client
        step = torch.where(ok, lp, zero)
        step = torch.where(impossible, torch.full_like(lp, NEG_INF), step)
        logp = logp + step
        if return_entropy:
            ent = ent + torch.where(ok, masked_entropy(lp_all, remaining), zero)
        taken = F.one_hot(safe, N).bool() & ok[:, None]
        remaining = remaining & ~taken
    return (logp, ent) if return_entropy else logp


def sample_k_without_replacement(
    logits: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample an ordered k-subset from the Plackett-Luce distribution.

    Uses the Gumbel top-k trick (exactly Plackett-Luce). With fewer than k
    eligible clients all eligible are selected and the rest is padded with -1.
    Deterministic mode: top-k of the logits among eligible.

    Returns:
        order[B,k] (long, -1 padded), logp[B], entropy[B] (path entropy, D8).
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B,N], got {tuple(logits.shape)}")
    B, N = logits.shape
    kk = min(int(k), N)
    scores = logits.detach()
    if not deterministic:
        scores = scores + gumbel_like(scores, generator)
    scores = scores.masked_fill(~mask, NEG_INF)
    top = scores.topk(kk, dim=1).indices
    valid = mask.gather(1, top)
    order = torch.where(valid, top, torch.full_like(top, -1))
    if kk < k:
        pad = torch.full((B, k - kk), -1, dtype=order.dtype, device=order.device)
        order = torch.cat([order, pad], dim=1)
    logp, ent = log_prob_of_order(logits, order, mask, return_entropy=True)
    return order, logp, ent
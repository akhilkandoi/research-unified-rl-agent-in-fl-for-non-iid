"""One self-attention layer across client embeddings (permutation-equivariant)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ClientAttention(nn.Module):
    """x -> LayerNorm(x + MHA(x,x,x)); identity when disabled.

    No positional encoding, so the layer is permutation-equivariant over the
    client axis and works for any N. Input/output: [B,N,H].
    """

    def __init__(self, dim: int, heads: int = 4, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        if self.enabled:
            if dim % heads != 0:
                raise ValueError(f"embedding dim {dim} must be divisible by attention heads {heads}")
            self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [B,N,H], got {tuple(x.shape)}")
        if not self.enabled:
            return x
        out, _ = self.mha(x, x, x, need_weights=False)
        return self.norm(x + out)
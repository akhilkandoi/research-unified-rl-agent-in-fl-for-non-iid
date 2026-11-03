"""Server aggregation with sample-count weights and optional server momentum."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from fedrl.utils.flat import weighted_average


def aggregate(vecs: torch.Tensor, n_samples: torch.Tensor, G: torch.Tensor,
              mom_state: Optional[torch.Tensor], m: float) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (G_new, mom_state).

    avg = sum_i n_i * w_i / sum_i n_i. If m == 0: G_new = avg exactly (state untouched).
    Else: delta = avg - G; v = m*v + delta; G_new = G + v.
    """
    w = torch.as_tensor(n_samples, dtype=vecs.dtype, device=vecs.device)
    if vecs.shape[0] != w.shape[0] or vecs.shape[0] == 0:
        raise ValueError(f"vecs {tuple(vecs.shape)} and n_samples {tuple(w.shape)} mismatch or empty")
    avg = weighted_average(vecs, w / w.sum())
    if m == 0:
        return avg, mom_state
    v_old = torch.zeros_like(G) if mom_state is None else mom_state
    v = m * v_old + (avg - G)
    return G + v, v
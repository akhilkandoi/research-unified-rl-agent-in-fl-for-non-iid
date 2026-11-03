"""Shared per-client encoders (same weights applied to every client).

Memory modes:
  none/ema/ema_delta : plain MLP on the state.
  gru_window         : GRU reads window [B,N,W,D]; MLP input = concat(state, GRU last output).
                       Recomputed from the window at PPO update time, so the GRU is trained.
  gru_const          : GRUCell(state, hidden[B,N,H]) -> new hidden; MLP input = concat(state, new hidden).
                       The incoming hidden is detached (D9): no backprop through time.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

MEMORY_MODES = ("none", "ema", "ema_delta", "gru_window", "gru_const")


def _init_linear(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Linear+ReLU stack; output dim = hidden[-1]."""

    def __init__(self, in_dim: int, hidden: Sequence[int]) -> None:
        super().__init__()
        if len(hidden) == 0:
            raise ValueError("policy.hidden must have at least one layer")
        layers = []
        d = in_dim
        for h in hidden:
            lin = nn.Linear(d, int(h))
            _init_linear(lin, math.sqrt(2.0))
            layers += [lin, nn.ReLU()]
            d = int(h)
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClientEncoder(nn.Module):
    """state[B,N,D] (+ window / hidden) -> embedding[B,N,E], new_hidden or None."""

    def __init__(self, state_dim: int, hidden: Sequence[int], memory_mode: str = "ema", gru_hidden: int = 64) -> None:
        super().__init__()
        if memory_mode not in MEMORY_MODES:
            raise ValueError(f"unknown state.memory_mode '{memory_mode}', expected one of {MEMORY_MODES}")
        self.mode = memory_mode
        self.state_dim = int(state_dim)
        self.gru_hidden = int(gru_hidden)
        in_dim = self.state_dim
        self.hidden_dim = 0  # size of the stored recurrent state (gru_const only)
        if memory_mode == "gru_window":
            self.gru = nn.GRU(self.state_dim, self.gru_hidden, batch_first=True)
            in_dim += self.gru_hidden
        elif memory_mode == "gru_const":
            self.cell = nn.GRUCell(self.state_dim, self.gru_hidden)
            in_dim += self.gru_hidden
            self.hidden_dim = self.gru_hidden
        self.mlp = MLP(in_dim, hidden)
        self.out_dim = self.mlp.out_dim

    def forward(
        self,
        state: torch.Tensor,
        window: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = state.shape
        new_hidden = None
        if self.mode == "gru_window":
            if window is None:
                raise ValueError("memory_mode=gru_window requires obs['window'] of shape [B,N,W,D]")
            W = window.shape[2]
            _, h = self.gru(window.reshape(B * N, W, D))
            feat = torch.cat([state, h[-1].reshape(B, N, self.gru_hidden)], dim=-1)
        elif self.mode == "gru_const":
            if hidden is None:
                raise ValueError("memory_mode=gru_const requires a hidden state [B,N,H] (obs['hidden'])")
            h_in = hidden.detach().reshape(B * N, self.gru_hidden)  # D9: constant at update time
            h_new = self.cell(state.reshape(B * N, D), h_in)
            new_hidden = h_new.reshape(B, N, self.gru_hidden)
            feat = torch.cat([state, new_hidden], dim=-1)
        else:
            feat = state
        return self.mlp(feat), new_hidden
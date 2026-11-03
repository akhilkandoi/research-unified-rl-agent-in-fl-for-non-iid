"""Per-client memory features for the environment (MASTER 2.7).

Modes: none (memory features zeroed), ema, ema_delta, gru_window (rolling window of the
last W states, zero padded at the start, current state is the last entry), gru_const
(no extra environment memory; the policy carries the hidden state).
Counters (idle, picked) are always maintained. EMA rules:
  loss EMA : initialised at the first probe, then beta*old + (1-beta)*scaled_loss,
             updated at EVERY probe observation (initial, post-round, drift re-probe).
  div  EMA : initialised at 0, updated only for clients picked this round.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from fedrl.env.state import EXTRA_FEATURE_NAMES, memory_mode, scale_div, scale_loss, state_dim

IndexLike = Union[Sequence[int], torch.Tensor]


class Memory:
    """Holds probe history, EMAs, counters, optional extra features and the state window."""

    def __init__(self, cfg, N: int, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.N = int(N)
        self.mode = memory_mode(cfg)
        self.T = int(cfg.fl.T)
        self.beta = float(cfg.state.ema_beta)
        self.W = int(cfg.state.gru_window)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.D = state_dim(cfg)
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Clear everything (episode start)."""
        z = lambda: torch.zeros(self.N, device=self.device)  # noqa: E731
        self.has_probe = False
        self.cur_loss: Optional[torch.Tensor] = None
        self.cur_acc: Optional[torch.Tensor] = None
        self.prev_loss: Optional[torch.Tensor] = None
        self.prev_acc: Optional[torch.Tensor] = None
        self.ema_loss, self.prev_ema_loss = z(), z()
        self.ema_div, self.prev_ema_div = z(), z()
        self.last_div = z()                     # raw relative divergence at last participation
        self.idle, self.picked = z(), z()
        self.cos = z()
        self.staleness = torch.ones(self.N, device=self.device)   # never trained -> 1
        self.global_acc = 0.0
        self.global_acc_change = 0.0
        self.comm_cum = 0.0
        self.comm_budget = 1.0
        self._window = (torch.zeros(self.N, self.W, self.D, device=self.device)
                        if self.mode == "gru_window" else None)

    def set_reference(self, global_acc: float = 0.0, comm_budget: float = 1.0) -> None:
        """Set the initial global accuracy and the communication budget (T * comm_ref)."""
        self.global_acc = float(global_acc)
        self.global_acc_change = 0.0
        self.comm_budget = max(float(comm_budget), 1e-12)

    # ------------------------------------------------------------------ updates
    def _idx(self, sel: IndexLike) -> torch.Tensor:
        return torch.as_tensor(list(sel) if not torch.is_tensor(sel) else sel,
                               dtype=torch.long, device=self.device).reshape(-1)

    def observe_probe(self, loss: torch.Tensor, acc: torch.Tensor) -> None:
        """Register a new probe of the global model on all clients."""
        loss = torch.nan_to_num(loss.detach().float().to(self.device), nan=0.0, posinf=1e6)
        acc = torch.nan_to_num(acc.detach().float().to(self.device), nan=0.0)
        self.prev_loss, self.prev_acc = self.cur_loss, self.cur_acc
        self.cur_loss, self.cur_acc = loss.clone(), acc.clone()
        scaled = scale_loss(loss, self.cfg)
        if not self.has_probe:
            self.ema_loss = scaled.clone()
            self.prev_ema_loss = scaled.clone()
            self.has_probe = True
        else:
            self.prev_ema_loss = self.ema_loss.clone()
            self.ema_loss = self.beta * self.ema_loss + (1.0 - self.beta) * scaled

    def record_round(
        self,
        selected: IndexLike,
        divergence: torch.Tensor,
        *,
        update_cos: Optional[torch.Tensor] = None,
        rounds_since_local: Optional[torch.Tensor] = None,
        global_acc: Optional[float] = None,
        global_acc_change: Optional[float] = None,
        comm: float = 0.0,
    ) -> None:
        """Update divergence memory, counters and optional extras after a round.

        ``update_cos`` is aligned with ``selected`` (cosine of each client's update with the
        aggregate update).
        """
        sel = self._idx(selected)
        self.prev_ema_div = self.ema_div.clone()
        self.idle = self.idle + 1.0
        if sel.numel() > 0:
            raw = torch.nan_to_num(divergence.detach().float().to(self.device), nan=0.0, posinf=1e6)[sel]
            self.last_div[sel] = raw
            self.ema_div[sel] = self.beta * self.ema_div[sel] + (1.0 - self.beta) * scale_div(raw, self.cfg)
            self.idle[sel] = 0.0
            self.picked[sel] = self.picked[sel] + 1.0
            if update_cos is not None:
                self.cos[sel] = torch.nan_to_num(update_cos.detach().float().to(self.device), nan=0.0)
        if rounds_since_local is not None:
            rs = torch.nan_to_num(rounds_since_local.detach().float().to(self.device),
                                  nan=float(self.T), posinf=float(self.T))
            self.staleness = (rs / self.T).clamp(0.0, 1.0)
        else:
            self.staleness = (self.idle / self.T).clamp(0.0, 1.0)
        if global_acc is not None:
            self.global_acc = float(global_acc)
        if global_acc_change is not None:
            self.global_acc_change = float(global_acc_change)
        self.comm_cum += float(comm)

    def push_state(self, state: torch.Tensor) -> None:
        """Append the current state to the rolling window (gru_window only)."""
        if self._window is None:
            return
        self._window = torch.roll(self._window, shifts=-1, dims=1)
        self._window[:, -1] = state.detach().to(self.device)

    # ------------------------------------------------------------------ features
    def features(self) -> torch.Tensor:
        """The 4 memory features [N,4]: ema loss, ema div, idle/T, picked/T (zeros if none)."""
        if self.mode == "none":
            return torch.zeros(self.N, 4, device=self.device)
        return torch.stack([
            self.ema_loss.clamp(0.0, 2.0),
            self.ema_div.clamp(0.0, 1.0),
            (self.idle / self.T).clamp(0.0, 1.0),
            (self.picked / self.T).clamp(0.0, 1.0),
        ], dim=1)

    def delta_features(self) -> torch.Tensor:
        """Change of the two EMAs since the previous observation [N,2] (ema_delta mode)."""
        return torch.stack([self.ema_loss - self.prev_ema_loss,
                            self.ema_div - self.prev_ema_div], dim=1).clamp(-1.0, 1.0)

    def extra_features(self) -> torch.Tensor:
        """Optional extra features [N,5] in EXTRA_FEATURE_NAMES order."""
        frac = min(max(self.comm_cum / self.comm_budget, 0.0), 1.0)
        cols = [
            self.cos.clamp(-1.0, 1.0),
            self.staleness,
            torch.full((self.N,), min(max(self.global_acc, 0.0), 1.0), device=self.device),
            torch.full((self.N,), min(max(self.global_acc_change, -1.0), 1.0), device=self.device),
            torch.full((self.N,), frac, device=self.device),
        ]
        out = torch.stack(cols, dim=1)
        assert out.shape[1] == len(EXTRA_FEATURE_NAMES)
        return out

    def window(self) -> Optional[torch.Tensor]:
        """[N,W,D] window of the last W states (gru_window), else None."""
        return None if self._window is None else self._window.clone()
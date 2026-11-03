"""Per-client state construction (MASTER 2.7).

Feature order (columns of the [N, D] state), base block = 24 numbers:
  0-9   label_prop_0..9   normalized label proportions (rows sum to 1)
  10    log_n             log(n_i) / log(clients.size_max), clip 0..1
  11    probe_loss        probe loss / state.loss_scale, clip 0..2
  12    probe_acc         probe accuracy
  13    d_probe_loss      (loss - previous probe loss) / loss_scale, clip -1..1
  14    d_probe_acc       acc - previous probe acc, clip -1..1 (0 on first probe)
  15    last_div          ||w_i-G||/||G|| at last participation / div_scale, clip 0..1
  16    log_speed         clip(ln(compute_speed), -log_clip, log_clip) / log_clip
  17    log_bandwidth     same for bandwidth
  18    battery           clip 0..1
  19    mem_ema_loss      EMA of scaled probe loss          (zeros if memory_mode == none)
  20    mem_ema_div       EMA of scaled divergence (picked) (zeros if none)
  21    mem_idle          idle rounds / T                    (zeros if none)
  22    mem_picked        times picked / T                   (zeros if none)
  23    round_frac        rounds completed / T
Then, only if memory_mode == ema_delta:
  mem_d_ema_loss, mem_d_ema_div  (change of the two EMAs since last observation, clip -1..1)
Then, only if state.extra_features:
  x_cos_update, x_staleness, x_global_acc, x_global_acc_change, x_comm_frac

All scaling constants are fixed (config state.*); no running statistics.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch

NUM_CLASSES = 10
MEMORY_MODES = ("none", "ema", "ema_delta", "gru_window", "gru_const")
LOSS_FEATURE_MAX = 2.0   # clip of the scaled probe loss (MASTER 2.7)
_EPS = 1e-8

BASE_FEATURE_NAMES: List[str] = (
    [f"label_prop_{c}" for c in range(NUM_CLASSES)]
    + ["log_n", "probe_loss", "probe_acc", "d_probe_loss", "d_probe_acc", "last_div",
       "log_speed", "log_bandwidth", "battery",
       "mem_ema_loss", "mem_ema_div", "mem_idle", "mem_picked", "round_frac"]
)
DELTA_FEATURE_NAMES: List[str] = ["mem_d_ema_loss", "mem_d_ema_div"]
EXTRA_FEATURE_NAMES: List[str] = [
    "x_cos_update", "x_staleness", "x_global_acc", "x_global_acc_change", "x_comm_frac",
]
# Full catalogue (maximum layout); use feature_names(cfg) for the active layout.
FEATURE_NAMES: List[str] = BASE_FEATURE_NAMES + DELTA_FEATURE_NAMES + EXTRA_FEATURE_NAMES


def memory_mode(cfg) -> str:
    """Return the validated memory mode."""
    mode = str(cfg.state.memory_mode)
    if mode not in MEMORY_MODES:
        raise ValueError(f"state.memory_mode must be one of {MEMORY_MODES}, got {mode!r}")
    return mode


def feature_names(cfg) -> List[str]:
    """Active feature names in column order for this config."""
    names = list(BASE_FEATURE_NAMES)
    if memory_mode(cfg) == "ema_delta":
        names += DELTA_FEATURE_NAMES
    if bool(cfg.state.extra_features):
        names += EXTRA_FEATURE_NAMES
    return names


def state_dim(cfg) -> int:
    """Number of state features per client (24 base, 26 ema_delta, +5 with extra_features)."""
    return len(feature_names(cfg))


def scale_loss(loss: torch.Tensor, cfg) -> torch.Tensor:
    """Fixed scaling of a probe loss: loss / loss_scale clipped to [0, 2]."""
    return (torch.nan_to_num(loss.float(), nan=0.0, posinf=1e6) / float(cfg.state.loss_scale)
            ).clamp(0.0, LOSS_FEATURE_MAX)


def scale_div(div: torch.Tensor, cfg) -> torch.Tensor:
    """Fixed scaling of a relative divergence: div / div_scale clipped to [0, 1]."""
    return (torch.nan_to_num(div.float(), nan=0.0, posinf=1e6) / float(cfg.state.div_scale)
            ).clamp(0.0, 1.0)


def _vec(t: torch.Tensor, n: int, name: str) -> torch.Tensor:
    t = t.float().reshape(-1)
    if t.numel() != n:
        raise ValueError(f"{name} must have {n} entries, got {t.numel()}")
    return torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)


def build_state(
    cfg,
    *,
    proportions: torch.Tensor,
    n: torch.Tensor,
    probe_loss: torch.Tensor,
    probe_acc: torch.Tensor,
    prev_loss: Optional[torch.Tensor],
    prev_acc: Optional[torch.Tensor],
    last_div: torch.Tensor,
    compute_speed: torch.Tensor,
    bandwidth: torch.Tensor,
    battery: torch.Tensor,
    mem_features: torch.Tensor,
    round_idx: int,
    mem_delta: Optional[torch.Tensor] = None,
    extras: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Assemble the [N, D] state. ``last_div`` is the RAW relative divergence (scaled here).

    ``mem_features`` [N,4] is ignored (zeroed) when memory_mode == none.
    ``mem_delta`` [N,2] is required for ema_delta, ``extras`` [N,5] when extra_features.
    """
    mode = memory_mode(cfg)
    N = int(proportions.shape[0])
    dev = proportions.device
    if proportions.shape != (N, NUM_CLASSES):
        raise ValueError(f"proportions must be [N,{NUM_CLASSES}], got {tuple(proportions.shape)}")
    props = torch.nan_to_num(proportions.float(), nan=0.0).clamp_min(0.0)
    props = props / props.sum(dim=1, keepdim=True).clamp_min(_EPS)
    log_n = (torch.log(_vec(n, N, "n").clamp_min(1.0)) / math.log(float(cfg.clients.size_max))
             ).clamp(0.0, 1.0)
    loss = _vec(probe_loss, N, "probe_loss")
    acc = _vec(probe_acc, N, "probe_acc").clamp(0.0, 1.0)
    if prev_loss is None or prev_acc is None:
        d_loss = torch.zeros(N, device=dev)
        d_acc = torch.zeros(N, device=dev)
    else:
        d_loss = ((loss - _vec(prev_loss, N, "prev_loss")) / float(cfg.state.loss_scale)).clamp(-1.0, 1.0)
        d_acc = (acc - _vec(prev_acc, N, "prev_acc").clamp(0.0, 1.0)).clamp(-1.0, 1.0)
    div = scale_div(_vec(last_div, N, "last_div"), cfg)
    lc = float(cfg.state.log_clip)
    log_speed = torch.log(_vec(compute_speed, N, "compute_speed").clamp_min(_EPS)).clamp(-lc, lc) / lc
    log_bw = torch.log(_vec(bandwidth, N, "bandwidth").clamp_min(_EPS)).clamp(-lc, lc) / lc
    batt = _vec(battery, N, "battery").clamp(0.0, 1.0)
    if mode == "none":
        mem = torch.zeros(N, 4, device=dev)
    else:
        if mem_features.shape != (N, 4):
            raise ValueError(f"mem_features must be [N,4], got {tuple(mem_features.shape)}")
        mem = mem_features.float().to(dev)
    rnd = torch.full((N,), min(max(float(round_idx) / float(cfg.fl.T), 0.0), 1.0), device=dev)

    cols = [props, log_n[:, None], scale_loss(loss, cfg)[:, None], acc[:, None], d_loss[:, None],
            d_acc[:, None], div[:, None], log_speed[:, None], log_bw[:, None], batt[:, None],
            mem, rnd[:, None]]
    if mode == "ema_delta":
        if mem_delta is None or mem_delta.shape != (N, len(DELTA_FEATURE_NAMES)):
            raise ValueError("ema_delta mode requires mem_delta of shape [N,2]")
        cols.append(mem_delta.float().to(dev).clamp(-1.0, 1.0))
    if bool(cfg.state.extra_features):
        if extras is None or extras.shape != (N, len(EXTRA_FEATURE_NAMES)):
            raise ValueError("extra_features requires extras of shape [N,5]")
        cols.append(extras.float().to(dev))
    state = torch.cat(cols, dim=1)
    if state.shape[1] != state_dim(cfg):
        raise RuntimeError(f"state width {state.shape[1]} != state_dim {state_dim(cfg)}")
    return state
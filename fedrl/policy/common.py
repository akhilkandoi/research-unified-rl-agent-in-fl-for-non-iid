"""Building blocks shared by JointPolicy and StackedPolicy.

The same ``score_*`` functions are used after sampling and in ``evaluate``,
so log-probs and entropies computed at sampling time and at PPO update time
are identical by construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from fedrl.policy.attention import ClientAttention
from fedrl.policy.encoder import ClientEncoder
from fedrl.policy.masking import categorical_logp_entropy, sample_categorical, sanitize_option_mask
from fedrl.policy.plackett_luce import log_prob_of_order, order_to_mask, sample_k_without_replacement

HEADS = ("select", "train", "mix")


def as_bool(v: Any) -> bool:
    """Robust bool (accepts strings such as 'false' coming from overrides)."""
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


@dataclass(frozen=True)
class MenuInfo:
    n_lr: int
    n_epochs: int
    n_batch: int
    n_alpha: int
    epochs_values: Tuple[float, ...]
    default_lr: int
    default_epochs: int
    default_batch: int
    default_alpha: int


def menu_info(cfg) -> MenuInfo:
    m = cfg.menus
    return MenuInfo(
        len(m.lr), len(m.epochs), len(m.batch), len(m.alpha),
        tuple(float(e) for e in m.epochs),
        int(m.default_lr_idx), int(m.default_epochs_idx), int(m.default_batch_idx), int(m.default_alpha_idx),
    )


def _init_head(lin: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(lin.weight, gain=gain)
    nn.init.zeros_(lin.bias)


# ----------------------------------------------------------------- modules
class PolicyBody(nn.Module):
    """Encoder + (optional) attention. Output embedding [B,N,E]."""

    def __init__(self, cfg, state_dim: int) -> None:
        super().__init__()
        hidden = [int(h) for h in cfg.policy.hidden]
        self.encoder = ClientEncoder(state_dim, hidden, str(cfg.state.memory_mode), int(cfg.policy.gru_hidden))
        self.out_dim = self.encoder.out_dim
        self.attention = ClientAttention(self.out_dim, int(cfg.policy.heads), as_bool(cfg.policy.attention))
        self.hidden_dim = self.encoder.hidden_dim

    def forward(self, state, window=None, hidden=None):
        emb, new_hidden = self.encoder(state, window, hidden)
        return self.attention(emb), new_hidden


class SelectHead(nn.Module):
    """1 logit per client -> [B,N]."""

    def __init__(self, emb_dim: int, info: MenuInfo) -> None:
        super().__init__()
        self.lin = nn.Linear(emb_dim, 1)
        _init_head(self.lin, 0.01)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.lin(emb).squeeze(-1)


class TrainHead(nn.Module):
    """3 categoricals per client: (lr, epochs, batch) logits, each [B,N,M]."""

    def __init__(self, emb_dim: int, info: MenuInfo) -> None:
        super().__init__()
        self.sizes = (info.n_lr, info.n_epochs, info.n_batch)
        self.lin = nn.Linear(emb_dim, sum(self.sizes))
        _init_head(self.lin, 0.01)

    def forward(self, emb: torch.Tensor):
        return torch.split(self.lin(emb), list(self.sizes), dim=-1)


class MixHead(nn.Module):
    """1 categorical per client over alpha -> [B,N,n_alpha]."""

    def __init__(self, emb_dim: int, info: MenuInfo) -> None:
        super().__init__()
        self.lin = nn.Linear(emb_dim, info.n_alpha)
        _init_head(self.lin, 0.01)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.lin(emb)


HEAD_CLASSES = {"select": SelectHead, "train": TrainHead, "mix": MixHead}


class ValueHead(nn.Module):
    """Mean-pooled client embeddings -> scalar value [B]."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(emb_dim, emb_dim)
        self.l2 = nn.Linear(emb_dim, 1)
        _init_head(self.l1, math.sqrt(2.0))
        _init_head(self.l2, 1.0)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.relu(self.l1(emb.mean(dim=1)))).squeeze(-1)


# ------------------------------------------------------------ obs / actions
def unsqueeze_obs(obs: Dict[str, Any]):
    """Unbatched obs -> batched tensors with B=1: state, eligible, epochs_mask, window, hidden."""
    state = obs["state"].unsqueeze(0)
    elig = obs["eligible"].bool().unsqueeze(0)
    emask = obs["epochs_mask"].bool().unsqueeze(0)
    window = obs.get("window")
    hidden = obs.get("hidden")
    return (
        state, elig, emask,
        None if window is None else window.unsqueeze(0),
        None if hidden is None else hidden.unsqueeze(0),
    )


def acts_from_batch(act_b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "order": act_b["selected"].long(),
        "lr_idx": act_b["lr_idx"].long(),
        "epochs_idx": act_b["epochs_idx"].long(),
        "batch_idx": act_b["batch_idx"].long(),
        "alpha_idx": act_b["alpha_idx"].long(),
    }


def pack_action(acts: Dict[str, torch.Tensor], b: int = 0) -> Dict[str, torch.Tensor]:
    order = acts["order"][b]
    return {
        "selected": order[order >= 0].long(),
        "lr_idx": acts["lr_idx"][b],
        "epochs_idx": acts["epochs_idx"][b],
        "batch_idx": acts["batch_idx"][b],
        "alpha_idx": acts["alpha_idx"][b],
    }


def reduce_chosen(per_client: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """Sum per-client [B,N] terms over chosen clients only -> [B]."""
    return torch.where(chosen, per_client, torch.zeros_like(per_client)).sum(dim=1)


# ----------------------------------------------------------------- defaults
def default_epochs_idx(epochs_mask: torch.Tensor, info: MenuInfo) -> torch.Tensor:
    """Default epochs option, or the largest feasible one if the default is masked."""
    vals = torch.tensor(info.epochs_values, device=epochs_mask.device)
    score = torch.where(epochs_mask, vals.expand(epochs_mask.shape), torch.full(epochs_mask.shape, -1.0, device=epochs_mask.device))
    largest = score.argmax(dim=-1)
    default = torch.full(epochs_mask.shape[:2], info.default_epochs, dtype=torch.long, device=epochs_mask.device)
    ok = epochs_mask[..., info.default_epochs]
    return torch.where(ok, default, largest)


def default_train_idx(epochs_mask: torch.Tensor, info: MenuInfo):
    shape = epochs_mask.shape[:2]
    dev = epochs_mask.device
    lr = torch.full(shape, info.default_lr, dtype=torch.long, device=dev)
    ba = torch.full(shape, info.default_batch, dtype=torch.long, device=dev)
    return lr, default_epochs_idx(epochs_mask, info), ba


# ----------------------------------------------------------------- sampling
def sample_select(logits, eligible, k, gen, deterministic):
    """Ordered picks [B,k] (-1 padded). logits None => inactive => uniform random among eligible."""
    if logits is None:
        zeros = torch.zeros(eligible.shape, device=eligible.device)
        order, _, _ = sample_k_without_replacement(zeros, eligible, k, gen, False)
        return order
    order, _, _ = sample_k_without_replacement(logits, eligible, k, gen, deterministic)
    return order


def sample_train(logits, epochs_mask, info, gen, deterministic):
    if logits is None:
        return default_train_idx(epochs_mask, info)
    lr_l, ep_l, ba_l = logits
    em = sanitize_option_mask(epochs_mask)
    return (
        sample_categorical(lr_l, None, gen, deterministic),
        sample_categorical(ep_l, em, gen, deterministic),
        sample_categorical(ba_l, None, gen, deterministic),
    )


def sample_mix(logits, like: torch.Tensor, info, gen, deterministic):
    if logits is None:
        return torch.full(like.shape, info.default_alpha, dtype=torch.long, device=like.device)
    return sample_categorical(logits, None, gen, deterministic)


def draw_actions(logits: Dict[str, Any], eligible, epochs_mask, info, k, gen, deterministic):
    lr, ep, ba = sample_train(logits.get("train"), epochs_mask, info, gen, deterministic)
    return {
        "order": sample_select(logits.get("select"), eligible, k, gen, deterministic),
        "lr_idx": lr, "epochs_idx": ep, "batch_idx": ba,
        "alpha_idx": sample_mix(logits.get("mix"), eligible, info, gen, deterministic),
    }


# ------------------------------------------------------------------ scoring
def score_actions(logits: Dict[str, Any], acts, eligible, epochs_mask) -> Dict[str, torch.Tensor]:
    """log-prob and entropy parts. Inactive heads (logits None) give exact zeros."""
    B, N = eligible.shape
    dev = eligible.device
    zeros_b = torch.zeros(B, device=dev)
    zeros_bn = torch.zeros(B, N, device=dev)
    order = acts["order"]
    chosen = order_to_mask(order, N)

    sel = logits.get("select")
    if sel is None:
        sel_lp, sel_ent = zeros_b, zeros_b.clone()
    else:
        sel_lp, sel_ent = log_prob_of_order(sel, order, eligible, return_entropy=True)

    tr = logits.get("train")
    if tr is None:
        tr_lp, tr_ent = zeros_bn, zeros_bn.clone()
    else:
        em = sanitize_option_mask(epochs_mask)
        l1, e1 = categorical_logp_entropy(tr[0], acts["lr_idx"], None)
        l2, e2 = categorical_logp_entropy(tr[1], acts["epochs_idx"], em)
        l3, e3 = categorical_logp_entropy(tr[2], acts["batch_idx"], None)
        tr_lp, tr_ent = l1 + l2 + l3, e1 + e2 + e3

    mx = logits.get("mix")
    if mx is None:
        mx_lp, mx_ent = zeros_bn, zeros_bn.clone()
    else:
        mx_lp, mx_ent = categorical_logp_entropy(mx, acts["alpha_idx"], None)

    return {"sel_lp": sel_lp, "sel_ent": sel_ent, "tr_lp": tr_lp, "tr_ent": tr_ent,
            "mx_lp": mx_lp, "mx_ent": mx_ent, "chosen": chosen}

import torch.nn as nn


class AgentDict(nn.ModuleDict):
    """ModuleDict that allows keys such as 'train' (which clash with nn.Module.train)."""

    _P = "m_"

    def __setitem__(self, key, module):
        super().__setitem__(self._P + key, module)

    def __getitem__(self, key):
        return super().__getitem__(self._P + key)

    def __delitem__(self, key):
        super().__delitem__(self._P + key)

    def __contains__(self, key):
        return super().__contains__(self._P + key)

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        return [k[len(self._P):] for k in super().keys()]

    def items(self):
        return [(k[len(self._P):], v) for k, v in super().items()]
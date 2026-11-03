"""Joint PPO policy: one network, three decision heads and one value head.

JOINT LOG-PROB = select log-prob (Plackett-Luce)
               + sum over CHOSEN clients of [log p(lr)+log p(epochs)+log p(batch)+log p(alpha)].
Entropy = selection path entropy (D8) + sum over chosen clients of the four
categorical entropies. Training/alpha are sampled for ALL clients but only
chosen clients count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from fedrl.policy.common import (
    HEAD_CLASSES, HEADS, PolicyBody, ValueHead, acts_from_batch, as_bool, draw_actions,
    menu_info, pack_action, reduce_chosen, score_actions, unsqueeze_obs, AgentDict
)


@dataclass
class PolicyOutput:
    """action: env action dict; logp/entropy/value: dicts keyed by agent name;
    hidden: NEW recurrent state [N,Htot] for gru_const (else None)."""
    action: Dict[str, torch.Tensor]
    logp: Dict[str, torch.Tensor]
    entropy: Dict[str, torch.Tensor]
    value: Dict[str, torch.Tensor]
    hidden: Optional[torch.Tensor] = None


class JointPolicy(nn.Module):
    """One policy for selection + training settings + mixing."""

    agent_names = ["joint"]

    def __init__(self, cfg, state_dim: int) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.k = int(cfg.fl.k)
        self.info = menu_info(cfg)
        self.memory_mode = str(cfg.state.memory_mode)
        self.active = {h: as_bool(cfg.get(f"policy.active_heads.{h}", True)) for h in HEADS}
        self.separate = as_bool(cfg.policy.separate_bodies)
        if self.separate:
            keys = ["select"] + [h for h in ("train", "mix") if self.active[h]]
        else:
            keys = ["shared"]
        self.bodies = AgentDict({key: PolicyBody(cfg, self.state_dim) for key in keys})
        emb_dim = self.bodies[keys[0]].out_dim
        self.heads = AgentDict({h: HEAD_CLASSES[h](emb_dim, self.info) for h in HEADS if self.active[h]})
        self.value_head = ValueHead(emb_dim)
        self._value_key = "select" if self.separate else "shared"
        self.hidden_per_body = int(cfg.policy.gru_hidden) if self.memory_mode == "gru_const" else 0
        self.hidden_dim = self.hidden_per_body * len(self.bodies)

    # ------------------------------------------------------------ helpers
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _body_key(self, head: str) -> str:
        return head if self.separate else "shared"

    def init_hidden(self, N: int) -> Optional[torch.Tensor]:
        """Zero GRU state [N,Htot] for gru_const, else None."""
        if self.hidden_dim == 0:
            return None
        return torch.zeros(N, self.hidden_dim, device=self._device())

    def _forward(self, state, window, hidden):
        """Batched forward: returns (logits dict, value[B], new_hidden[B,N,Htot]|None)."""
        embs: Dict[str, torch.Tensor] = {}
        parts = []
        off, Hb = 0, self.hidden_per_body
        for key, body in self.bodies.items():
            h = None
            if Hb:
                if hidden is None:
                    raise ValueError("memory_mode=gru_const requires obs['hidden'] [B,N,Htot]")
                if hidden.shape[-1] != self.hidden_dim:
                    raise ValueError(f"hidden last dim {hidden.shape[-1]} != expected {self.hidden_dim}")
                h = hidden[..., off:off + Hb]
                off += Hb
            emb, nh = body(state, window, h)
            embs[key] = emb
            if nh is not None:
                parts.append(nh)
        logits = {h: self.heads[h](embs[self._body_key(h)]) for h in HEADS if h in self.heads}
        value = self.value_head(embs[self._value_key])
        new_hidden = torch.cat(parts, dim=-1) if parts else None
        return logits, value, new_hidden

    @staticmethod
    def _combine(parts) -> Tuple[torch.Tensor, torch.Tensor]:
        ch = parts["chosen"]
        logp = parts["sel_lp"] + reduce_chosen(parts["tr_lp"] + parts["mx_lp"], ch)
        ent = parts["sel_ent"] + reduce_chosen(parts["tr_ent"] + parts["mx_ent"], ch)
        return logp, ent

    # ------------------------------------------------------------ public API
    @torch.no_grad()
    def sample(self, obs: Dict[str, Any], gen: Optional[torch.Generator] = None,
               deterministic: bool = False) -> PolicyOutput:
        """Act on one unbatched observation (runs without grad; outputs detached)."""
        state, elig, emask, window, hidden = unsqueeze_obs(obs)
        if self.hidden_dim and hidden is None:
            hidden = self.init_hidden(state.shape[1]).unsqueeze(0)
        logits, value, new_hidden = self._forward(state, window, hidden)
        acts = draw_actions(logits, elig, emask, self.info, self.k, gen, deterministic)
        parts = score_actions(logits, acts, elig, emask)
        logp, ent = self._combine(parts)
        return PolicyOutput(
            action=pack_action(acts, 0),
            logp={"joint": logp[0]}, entropy={"joint": ent[0]}, value={"joint": value[0]},
            hidden=None if new_hidden is None else new_hidden[0],
        )

    def evaluate(self, obs_b: Dict[str, Any], act_b: Dict[str, torch.Tensor]):
        """Recompute (logp, entropy, value) dicts [B] for stored batched actions."""
        elig = obs_b["eligible"].bool()
        emask = obs_b["epochs_mask"].bool()
        logits, value, _ = self._forward(obs_b["state"], obs_b.get("window"), obs_b.get("hidden"))
        parts = score_actions(logits, acts_from_batch(act_b), elig, emask)
        logp, ent = self._combine(parts)
        return {"joint": logp}, {"joint": ent}, {"joint": value}
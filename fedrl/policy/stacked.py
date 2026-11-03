"""Three independent single-head agents (selection / training / alpha).

Each agent has its own encoder, attention, head, value head (no shared
parameters), sees only the state (never the other agents' actions) and is
trained by its own PPO optimizer. Log-probs of the train and mix agents count
chosen clients only (the chosen set is used for credit accounting, not as input).
`policy.active_heads` and `policy.separate_bodies` are ignored here.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from fedrl.policy.common import (
    HEAD_CLASSES, HEADS, PolicyBody, ValueHead, acts_from_batch, draw_actions,
    menu_info, pack_action, reduce_chosen, score_actions, unsqueeze_obs, AgentDict
)
from fedrl.policy.joint_policy import JointPolicy, PolicyOutput


class _SingleHeadAgent(nn.Module):
    def __init__(self, cfg, state_dim: int, kind: str, info) -> None:
        super().__init__()
        self.kind = kind
        self.body = PolicyBody(cfg, state_dim)
        self.head = HEAD_CLASSES[kind](self.body.out_dim, info)
        self.value_head = ValueHead(self.body.out_dim)

    def forward(self, state, window, hidden):
        emb, nh = self.body(state, window, hidden)
        return self.head(emb), self.value_head(emb), nh


class StackedPolicy(nn.Module):
    """Three independent agents with the same architecture as the joint policy."""

    agent_names = ["select", "train", "mix"]

    def __init__(self, cfg, state_dim: int) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.k = int(cfg.fl.k)
        self.info = menu_info(cfg)
        self.memory_mode = str(cfg.state.memory_mode)
        self.agents = AgentDict({n: _SingleHeadAgent(cfg, self.state_dim, n, self.info) for n in HEADS})
        self.hidden_per_agent = int(cfg.policy.gru_hidden) if self.memory_mode == "gru_const" else 0
        self.hidden_dim = self.hidden_per_agent * len(self.agents)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def init_hidden(self, N: int) -> Optional[torch.Tensor]:
        if self.hidden_dim == 0:
            return None
        return torch.zeros(N, self.hidden_dim, device=self._device())

    def _forward(self, state, window, hidden):
        logits, values, parts = {}, {}, []
        off, Hb = 0, self.hidden_per_agent
        for name, agent in self.agents.items():
            h = None
            if Hb:
                if hidden is None:
                    raise ValueError("memory_mode=gru_const requires obs['hidden'] [B,N,Htot]")
                if hidden.shape[-1] != self.hidden_dim:
                    raise ValueError(f"hidden last dim {hidden.shape[-1]} != expected {self.hidden_dim}")
                h = hidden[..., off:off + Hb]
                off += Hb
            lg, v, nh = agent(state, window, h)
            logits[name], values[name] = lg, v
            if nh is not None:
                parts.append(nh)
        return logits, values, (torch.cat(parts, dim=-1) if parts else None)

    @staticmethod
    def _split(parts):
        ch = parts["chosen"]
        logp = {"select": parts["sel_lp"], "train": reduce_chosen(parts["tr_lp"], ch),
                "mix": reduce_chosen(parts["mx_lp"], ch)}
        ent = {"select": parts["sel_ent"], "train": reduce_chosen(parts["tr_ent"], ch),
               "mix": reduce_chosen(parts["mx_ent"], ch)}
        return logp, ent

    @torch.no_grad()
    def sample(self, obs: Dict[str, Any], gen: Optional[torch.Generator] = None,
               deterministic: bool = False) -> PolicyOutput:
        state, elig, emask, window, hidden = unsqueeze_obs(obs)
        if self.hidden_dim and hidden is None:
            hidden = self.init_hidden(state.shape[1]).unsqueeze(0)
        logits, values, new_hidden = self._forward(state, window, hidden)
        acts = draw_actions(logits, elig, emask, self.info, self.k, gen, deterministic)
        logp, ent = self._split(score_actions(logits, acts, elig, emask))
        return PolicyOutput(
            action=pack_action(acts, 0),
            logp={n: v[0] for n, v in logp.items()},
            entropy={n: v[0] for n, v in ent.items()},
            value={n: v[0] for n, v in values.items()},
            hidden=None if new_hidden is None else new_hidden[0],
        )

    def evaluate(self, obs_b: Dict[str, Any], act_b: Dict[str, torch.Tensor]):
        elig = obs_b["eligible"].bool()
        emask = obs_b["epochs_mask"].bool()
        logits, values, _ = self._forward(obs_b["state"], obs_b.get("window"), obs_b.get("hidden"))
        logp, ent = self._split(score_actions(logits, acts_from_batch(act_b), elig, emask))
        return logp, ent, values


def count_parameters(policy: nn.Module) -> Dict[str, Any]:
    """Parameter counts: {'total': int, 'by_module': {name: int}} (per agent for stacked)."""
    def n(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    if hasattr(policy, "agents"):
        by = {f"agent:{name}": n(a) for name, a in policy.agents.items()}
    else:
        by = {name: n(m) for name, m in policy.named_children()}
    return {"total": n(policy), "by_module": by}


def print_parameter_report(cfg, state_dim: int) -> Dict[str, Any]:
    """Print (and return) parameter counts of joint vs stacked for this config."""
    rep = {"joint": count_parameters(JointPolicy(cfg, state_dim)),
           "stacked": count_parameters(StackedPolicy(cfg, state_dim))}
    lines = [f"state_dim={state_dim}  hidden={list(cfg.policy.hidden)}  attention={cfg.policy.attention}  "
             f"memory_mode={cfg.state.memory_mode}"]
    for name, r in rep.items():
        lines.append(f"{name:8s} total parameters: {r['total']:>9,d}")
        for m, c in r["by_module"].items():
            lines.append(f"    {m:16s} {c:>9,d}")
    ratio = rep["stacked"]["total"] / max(rep["joint"]["total"], 1)
    lines.append(f"stacked / joint parameter ratio: {ratio:.2f}")
    print("\n".join(lines))
    return rep
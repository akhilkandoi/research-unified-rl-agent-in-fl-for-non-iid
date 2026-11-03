"""Reward computation and per-agent routing (MASTER 2.9, 2.8).

Terms (validation data only):
  r_global = acc_scale * (global val acc after - before)
  r_pers   = acc_scale * pers_gain_chosen   (blended minus local-only accuracy, chosen clients)
  r_fair   = acc_scale * (p10 after - p10 before)
  comm_norm, comp_norm = clip(cost / reference, 0, cost_clip)  (via fedrl.fl.costs.normalize_costs)
  final_bonus (last round only) = cfg.reward.final_bonus * final global val accuracy
  reward = w_global*r_global + w_pers*r_pers + w_fair*r_fair - w_comm*comm_norm
           - w_comp*comp_norm + final_bonus
Routing design choice for the final bonus: shared mode gives it to every agent; specialized
mode gives w_global*final_bonus to selection and training, and nothing to mix.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from fedrl.fl.costs import normalize_costs

JOINT_AGENT = "joint"
STACKED_AGENTS = ("select", "train", "mix")
ROUTING_MODES = ("joint", "stacked_shared", "stacked_specialized")


def _finite(x: Any, default: float = 0.0) -> float:
    """float(x) if finite else default (empty selections can yield NaN means)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def total_from_terms(terms: Dict[str, float], cfg) -> float:
    """Weighted total reward from a terms dict."""
    rc = cfg.reward
    return (float(rc.w_global) * terms["r_global"] + float(rc.w_pers) * terms["r_pers"]
            + float(rc.w_fair) * terms["r_fair"] - float(rc.w_comm) * terms["comm_norm"]
            - float(rc.w_comp) * terms["comp_norm"] + terms["final_bonus"])


def compute_reward(result, refs: Tuple[float, float], cfg, last_round: bool) -> Tuple[float, Dict[str, float]]:
    """Reward and all terms for one round. ``refs`` = (comm_ref, comp_ref)."""
    rc = cfg.reward
    acc_scale = float(rc.acc_scale)
    r_global = acc_scale * (_finite(result.global_val_acc_after) - _finite(result.global_val_acc_before))
    r_pers = acc_scale * _finite(result.pers_gain_chosen)
    r_fair = acc_scale * (_finite(result.fair_p10_after) - _finite(result.fair_p10_before))
    comm_norm, comp_norm = normalize_costs(_finite(result.comm), _finite(result.comp), refs, cfg)
    comm_norm, comp_norm = float(comm_norm), float(comp_norm)
    bonus = float(rc.final_bonus) * _finite(result.global_val_acc_after) if last_round else 0.0
    terms = {
        "r_global": r_global, "r_pers": r_pers, "r_fair": r_fair,
        "comm_norm": comm_norm, "comp_norm": comp_norm, "final_bonus": bonus,
    }
    total = total_from_terms(terms, cfg)
    terms["total"] = total
    return total, terms


def route_rewards(terms: Dict[str, float], cfg, mode: str) -> Dict[str, float]:
    """Per-agent rewards for joint / stacked_shared / stacked_specialized."""
    rc = cfg.reward
    total = float(terms["total"]) if "total" in terms else total_from_terms(terms, cfg)
    if mode == "joint":
        return {JOINT_AGENT: total}
    if mode == "stacked_shared":
        return {a: total for a in STACKED_AGENTS}
    if mode == "stacked_specialized":
        wg = float(rc.w_global)
        bonus = wg * terms["final_bonus"]
        return {
            "select": wg * terms["r_global"] + float(rc.w_fair) * terms["r_fair"] + bonus,
            "train": wg * terms["r_global"] - float(rc.w_comp) * terms["comp_norm"] + bonus,
            "mix": float(rc.w_pers) * terms["r_pers"],
        }
    raise ValueError(f"unknown routing mode {mode!r}; expected one of {ROUTING_MODES}")
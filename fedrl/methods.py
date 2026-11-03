"""Method registry (MASTER 2.12): name -> kind, config overrides, reward mode."""
from __future__ import annotations

import copy
import re
from typing import Any

HEADS = ("select", "train", "mix")


def _heads(select: bool, train: bool, mix: bool) -> dict[str, Any]:
    return {
        "policy.active_heads.select": select,
        "policy.active_heads.train": train,
        "policy.active_heads.mix": mix,
    }


def _rl(kind: str, overrides: dict[str, Any], reward_mode: str) -> dict[str, Any]:
    ov = dict(overrides)
    ov["reward.mode"] = reward_mode
    return {"kind": kind, "overrides": ov, "reward_mode": reward_mode}


def _joint(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return _rl("rl_joint", overrides or {}, "joint")


def _base(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"kind": "baseline", "overrides": dict(overrides or {}), "reward_mode": None}


METHODS: dict[str, dict[str, Any]] = {
    # RL
    "joint": _joint(),
    "stacked_shared": _rl("rl_stacked", {}, "stacked_shared"),
    "stacked_specialized": _rl("rl_stacked", {}, "stacked_specialized"),
    "single_select": _joint(_heads(True, False, False)),
    "single_train": _joint(_heads(False, True, False)),
    "single_mix": _joint(_heads(False, False, True)),
    # ablations (core)
    "joint_mem_none": _joint({"state.memory_mode": "none"}),
    "joint_no_select": _joint(_heads(False, True, True)),
    "joint_no_train": _joint(_heads(True, False, True)),
    "joint_no_mix": _joint(_heads(True, True, False)),
    "joint_gru_window": _joint({"state.memory_mode": "gru_window"}),
    # ablations (extended)
    "joint_ema_beta0.5": _joint({"state.ema_beta": 0.5}),
    "joint_ema_delta": _joint({"state.memory_mode": "ema_delta"}),
    "joint_separate": _joint({"policy.separate_bodies": True}),
    "joint_no_attention": _joint({"policy.attention": False}),
    "joint_extra_features": _joint({"state.extra_features": True}),
    "joint_gru_const": _joint({"state.memory_mode": "gru_const"}),
    # non-RL baselines (controllers implemented in Part 7)
    "fedavg": _base(),
    "fixed_alpha_0.5": _base(),
    "fixed_alpha_0.1": _base(),
    "fedprox": _base({"fl.prox_mu": 0.01}),
    "loss_heuristic": _base(),
    "tuned_settings": _base(),
    "tuned_alpha": _base(),
    "bandit": _base(),
    "value_selector": _base(),
    "adaptive_mixing": _base(),
}


def all_method_names() -> list[str]:
    """All registered method names."""
    return list(METHODS.keys())


def is_rl(name: str) -> bool:
    """True for PPO-trained methods (joint or stacked)."""
    return _get(name)["kind"] in ("rl_joint", "rl_stacked")


def _get(name: str) -> dict[str, Any]:
    if name not in METHODS:
        raise KeyError(f"Unknown method '{name}'. Known: {sorted(METHODS)}")
    return METHODS[name]


def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        nxt = d.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            d[p] = nxt
        d = nxt
    d[parts[-1]] = value


def _setattr_dotted(obj: Any, key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def clone_config(cfg: Any, overrides: dict[str, Any]):
    """Return a NEW Config equal to cfg with dotted-key overrides applied (cfg untouched)."""
    d = copy.deepcopy(cfg.to_dict())
    for k, v in overrides.items():
        _set_dotted(d, k, copy.deepcopy(v))
    try:
        return type(cfg)(d)
    except TypeError:
        new = copy.deepcopy(cfg)
        for k, v in overrides.items():
            _setattr_dotted(new, k, copy.deepcopy(v))
        return new


def apply_method(cfg: Any, name: str):
    """Return a new Config with the method's overrides applied; input is not mutated."""
    return clone_config(cfg, _get(name)["overrides"])


_VARIANT_RE = re.compile(r"^(w(comm|fair|comp|pers|global)|upd)(-?\d+(?:\.\d+)?)$")
_KEYMAP = {"comm": "reward.w_comm", "fair": "reward.w_fair", "comp": "reward.w_comp",
           "pers": "reward.w_pers", "global": "reward.w_global"}


def variant_overrides(variant: str, strict: bool = True) -> dict[str, Any]:
    """Parse variants such as 'default', 'wcomm0', 'wcomm0.3', 'wfair0', 'wcomm0_wfair0.3', 'upd4'.

    strict=False ignores free-form run labels (e.g. 'straight'); malformed reward/upd tokens still raise.
    """
    if variant in ("", "default"):
        return {}
    out: dict[str, Any] = {}
    for tok in variant.split("_"):
        m = _VARIANT_RE.match(tok)
        if not m:
            if strict or re.match(r"^(w(comm|fair|comp|pers|global)|upd)", tok):
                raise ValueError(f"Cannot parse variant token '{tok}' in '{variant}'. "
                                 "Use e.g. wcomm0, wfair0.3, upd4, joined with '_'.")
            continue  # free-form run label
        if m.group(1) == "upd":
            out["ppo.update_every_episodes"] = int(float(m.group(3)))
        else:
            out[_KEYMAP[m.group(2)]] = float(m.group(3))
    return out


def apply_variant(cfg: Any, variant: str, strict: bool = True):
    """Return a new Config with the variant's overrides applied."""
    return clone_config(cfg, variant_overrides(variant, strict))
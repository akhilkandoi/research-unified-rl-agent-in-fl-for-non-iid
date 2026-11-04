"""Controller base class and shared helpers for building valid actions.

All controllers (baselines and RL) return the action dict described in
MASTER 2.6 and respect the same eligibility and epochs masks as the policy.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

import fedrl.env.state as _state_module
from fedrl.utils.logging_utils import get_logger
from abc import ABC, abstractmethod

LOGGER = get_logger(__name__)

FEATURE_NAMES = getattr(_state_module, "FEATURE_NAMES", None)
_ALLOWED_OVERRIDES = ("lr_value", "alpha_value", "prox_mu")


def feature_index(candidates: Sequence[str], default: int) -> int:
    """Index of the first candidate name found in FEATURE_NAMES, else ``default``."""
    names = list(FEATURE_NAMES) if FEATURE_NAMES is not None else []
    for cand in candidates:
        if cand in names:
            return names.index(cand)
    return default


# Layout (MASTER 2.7): data 0-10, perf 11-14, divergence 15, resources 16-18 ...
PROBE_LOSS_IDX = feature_index(("probe_loss", "loss", "probe_loss_scaled"), 11)
COMPUTE_SPEED_IDX = feature_index(("log_speed", "compute_speed", "log_compute_speed", "speed"), 16)


class Controller(ABC):
    """Base class for every method that outputs actions for the FL environment."""

    name: str = "controller"

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.k: int = int(cfg.fl.k)

    def reset(self, obs: dict) -> None:
        """Called once at the start of every episode."""

    @abstractmethod
    def act(self, obs: dict, rng: np.random.Generator, deterministic: bool = True) -> dict:
        """Return an action dict for the current observation."""

    def observe(self, obs: dict, action: dict, reward: float, next_obs: dict | None,
                info: dict) -> None:
        """Learning controllers override this; default is a no-op."""

    def bind_env(self, env: Any, data: tuple | None = None) -> None:
        """Optional hook giving a controller evaluation-side access to the env."""



def _np_bool(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(bool)
    return np.asarray(x, dtype=bool)


def _expand_idx(v: Any, default: int, n: int, device: torch.device, upper: int,
                what: str) -> torch.Tensor:
    """Turn a scalar / sequence / tensor into a LongTensor[N] of menu indices."""
    if v is None:
        v = default
    t = torch.as_tensor(v, dtype=torch.long).to(device)
    if t.ndim == 0:
        t = t.repeat(n)
    if tuple(t.shape) != (n,):
        raise ValueError(f"{what} must be a scalar or have shape ({n},), got {tuple(t.shape)}")
    if bool((t < 0).any()) or bool((t >= upper).any()):
        raise ValueError(f"{what} contains indices outside [0, {upper - 1}]")
    return t.clone()


def _value_vec(v: Any, n: int, device: torch.device, what: str) -> torch.Tensor:
    t = torch.as_tensor(v, dtype=torch.float32).to(device)
    if t.ndim == 0:
        t = t.repeat(n)
    if tuple(t.shape) != (n,):
        raise ValueError(f"{what} must be a scalar or have shape ({n},), got {tuple(t.shape)}")
    return t.clone()


def apply_epochs_mask(epochs_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked epochs options by the largest feasible option (per client)."""
    mask = mask.to(epochs_idx.device).bool()
    ar = torch.arange(mask.shape[1], device=mask.device)
    best = torch.where(mask, ar, torch.full_like(ar, -1)).max(dim=1).values
    cur_ok = mask.gather(1, epochs_idx[:, None]).squeeze(1)
    repl = torch.where(best >= 0, best, epochs_idx)
    return torch.where(cur_ok, epochs_idx, repl)


def make_action(selected: Any, lr_idx: Any, epochs_idx: Any, batch_idx: Any, alpha_idx: Any,
                obs: dict, cfg: Any, **overrides: Any) -> dict:
    """Build a valid action dict for all N clients.

    ``None`` index arguments use menus.default_*_idx. Selected clients are
    de-duplicated, filtered by eligibility and truncated to ``fl.k``. Masked epochs
    options fall back to the largest feasible option. Supported overrides:
    ``lr_value``, ``alpha_value`` (scalar or [N]) and ``prox_mu``.
    """
    bad = [key for key in overrides if key not in _ALLOWED_OVERRIDES]
    if bad:
        raise TypeError(f"unknown action overrides {bad}; allowed: {_ALLOWED_OVERRIDES}")
    state = obs["state"]
    device, n = state.device, int(state.shape[0])
    menus = cfg.menus
    eligible = _np_bool(obs["eligible"])

    sel: list[int] = []
    for i in torch.as_tensor(selected).reshape(-1).cpu().tolist():
        i = int(i)
        if 0 <= i < n and eligible[i] and i not in sel:
            sel.append(i)
    sel = sel[: int(cfg.fl.k)]

    e_idx = _expand_idx(epochs_idx, menus.default_epochs_idx, n, device, len(menus.epochs),
                        "epochs_idx")
    if obs.get("epochs_mask") is not None:
        e_idx = apply_epochs_mask(e_idx, obs["epochs_mask"].to(device))
    action: dict = {
        "selected": torch.as_tensor(sel, dtype=torch.long, device=device),
        "lr_idx": _expand_idx(lr_idx, menus.default_lr_idx, n, device, len(menus.lr), "lr_idx"),
        "epochs_idx": e_idx,
        "batch_idx": _expand_idx(batch_idx, menus.default_batch_idx, n, device,
                                 len(menus.batch), "batch_idx"),
        "alpha_idx": _expand_idx(alpha_idx, menus.default_alpha_idx, n, device,
                                 len(menus.alpha), "alpha_idx"),
    }
    for key in ("lr_value", "alpha_value"):
        if overrides.get(key) is not None:
            action[key] = _value_vec(overrides[key], n, device, key)
    if overrides.get("prox_mu") is not None:
        action["prox_mu"] = float(overrides["prox_mu"])
    return action


def random_eligible_selection(obs: dict, k: int, rng: np.random.Generator) -> torch.Tensor:
    """Choose min(k, #eligible) eligible clients uniformly without replacement."""
    device = obs["state"].device
    elig = np.flatnonzero(_np_bool(obs["eligible"]))
    m = min(int(k), elig.size)
    if m == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    pick = rng.choice(elig, size=m, replace=False)
    return torch.as_tensor(pick, dtype=torch.long, device=device)


def top_k_eligible(obs: dict, k: int, *keys: np.ndarray) -> torch.Tensor:
    """Top-k eligible clients by descending ``keys[0]``, then ``keys[1]`` ..., then index."""
    device = obs["state"].device
    elig = np.flatnonzero(_np_bool(obs["eligible"]))
    m = min(int(k), elig.size)
    if m == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    cols = []
    for key in keys:
        arr = np.asarray(key, dtype=np.float64)[elig]
        cols.append(-np.where(np.isnan(arr), -np.inf, arr))
    order = np.lexsort((elig,) + tuple(reversed(cols)))
    return torch.as_tensor(elig[order[:m]], dtype=torch.long, device=device)


def synthetic_obs(cfg: Any, state_dim: int, rng: np.random.Generator, device: Any = "cpu",
                  n_ineligible: int = 0, mask_top_epochs: bool = False,
                  round_t: int = 1) -> dict:
    """Random observation with the real obs layout (for tests and toy problems)."""
    n = int(cfg.fl.N)
    n_ep = len(cfg.menus.epochs)
    state = torch.as_tensor(rng.standard_normal((n, state_dim)), dtype=torch.float32,
                            device=device)
    eligible = np.ones(n, dtype=bool)
    if n_ineligible:
        eligible[rng.choice(n, size=min(n_ineligible, n), replace=False)] = False
    mask = np.ones((n, n_ep), dtype=bool)
    mask[~eligible] = False
    if mask_top_epochs:
        mask[np.arange(n) % 2 == 0, n_ep - 1] = False
    return {"state": state,
            "eligible": torch.as_tensor(eligible, device=device),
            "epochs_mask": torch.as_tensor(mask, device=device),
            "round": int(round_t)}

def rollout_episode(env: Any, controller: Controller, spec: Any, rng: np.random.Generator,
                    deterministic: bool = True, learn: bool = False,
                    data: tuple | None = None) -> dict:
    """Run one full FL episode; returns return, rounds, final val accuracy."""
    obs = env.reset(spec)            # env.reset builds a NEW world, so bind AFTER reset
    controller.reset(obs)
    controller.bind_env(env, data)
    total, rounds, info = 0.0, 0, {}
    done = False
    while not done:
        action = controller.act(obs, rng, deterministic=deterministic)
        next_obs, reward, done, info = env.step(action)
        if learn:
            controller.observe(obs, action, reward, next_obs, {**info, "done": done})
        total += float(reward)
        rounds += 1
        obs = next_obs
    return {"return": total, "rounds": rounds, "final_val_acc": info.get("global_val_acc")}
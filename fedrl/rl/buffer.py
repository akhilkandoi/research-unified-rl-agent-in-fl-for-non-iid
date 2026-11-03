"""Rollout buffer with per-agent GAE, advantage normalization and minibatching."""
from __future__ import annotations

from typing import Any, Callable, Iterator

import numpy as np
import torch

ACTION_KEYS = ("lr_idx", "epochs_idx", "batch_idx", "alpha_idx")


def compute_gae(rewards, values, dones, gamma: float, lam: float, last_value: float = 0.0):
    """GAE for one episode. Returns (advantages, returns); value after a done step is 0."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    last = 0.0
    for t in reversed(range(T)):
        nonterm = 0.0 if dones[t] else 1.0
        next_v = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


from fedrl.env.reward import route_rewards as _env_route_rewards  # Part 4 (authoritative)

TERM_KEYS = ("r_global", "r_pers", "r_fair", "comm_norm", "comp_norm", "final_bonus", "total")


def step_rewards(info: dict, cfg: Any, agent_names: list[str]) -> dict[str, float]:
    """Per-agent rewards for one env step using Part 4's routing (never re-implemented here).
    joint policy -> {"joint": total}; stacked -> mode taken from cfg.reward.mode."""
    terms = {k: float(info[k]) for k in TERM_KEYS}
    mode = "joint" if list(agent_names) == ["joint"] else cfg.reward.mode
    routed = _env_route_rewards(terms, cfg, mode)
    missing = set(agent_names) - set(routed)
    if missing:
        raise KeyError(f"Reward routing mode '{mode}' returned no reward for agents {sorted(missing)}")
    return {a: float(routed[a]) for a in agent_names}

def _c(x):
    return None if x is None else x.detach().clone()


class RolloutBuffer:
    """Stores transitions of whole episodes; GAE per agent; yields minibatches."""

    def __init__(self, cfg: Any, agent_names: list[str]):
        self.agent_names = list(agent_names)
        self.k = int(cfg.fl.k)
        self.gamma = float(cfg.ppo.gamma)
        self.lam = float(cfg.ppo.gae_lambda)
        self.n_minibatches = int(cfg.ppo.minibatches)
        self.clear()

    def clear(self) -> None:
        self._steps: list[dict] = []
        self._ep_start = 0
        self._adv = {a: [] for a in self.agent_names}
        self._ret = {a: [] for a in self.agent_names}
        self._cache: dict | None = None

    def __len__(self) -> int:
        return len(self._steps)

    def add(self, obs: dict, out: Any, rewards: dict[str, float], done: bool) -> None:
        """Store one step. obs may contain 'hidden' (GRU input). out is a PolicyOutput."""
        act = out.action
        sel = act["selected"].long().reshape(-1)[: self.k]
        padded = torch.full((self.k,), -1, dtype=torch.long, device=sel.device)
        padded[: sel.numel()] = sel
        step = {
            "state": _c(obs["state"]), "eligible": _c(obs["eligible"]),
            "epochs_mask": _c(obs["epochs_mask"]), "window": _c(obs.get("window")),
            "hidden": _c(obs.get("hidden")), "selected": padded,
            "logp": {a: float(out.logp[a]) for a in self.agent_names},
            "value": {a: float(out.value[a]) for a in self.agent_names},
            "reward": {a: float(rewards[a]) for a in self.agent_names},
            "done": bool(done),
        }
        for key in ACTION_KEYS:
            step[key] = act[key].detach().long().clone()
        self._steps.append(step)
        self._cache = None

    def finish_episode(self, last_value: float | dict = 0.0,
                       reward_fns: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None) -> None:
        """Compute GAE for steps added since the last call (value 0 after a final done step)."""
        s, e = self._ep_start, len(self._steps)
        if e == s:
            return
        steps = self._steps[s:e]
        dones = [st["done"] for st in steps]
        for a in self.agent_names:
            r = np.array([st["reward"][a] for st in steps], dtype=np.float64)
            if reward_fns and a in reward_fns:
                r = np.asarray(reward_fns[a](r), dtype=np.float64)
            v = np.array([st["value"][a] for st in steps], dtype=np.float64)
            lv = last_value[a] if isinstance(last_value, dict) else last_value
            adv, ret = compute_gae(r, v, dones, self.gamma, self.lam, float(lv))
            self._adv[a].extend(adv.tolist())
            self._ret[a].extend(ret.tolist())
        self._ep_start = e
        self._cache = None

    def raw_advantages(self, agent: str) -> np.ndarray:
        return np.asarray(self._adv[agent])

    def returns(self, agent: str) -> np.ndarray:
        return np.asarray(self._ret[agent])

    def _build(self) -> dict:
        st = self._steps
        dev = st[0]["state"].device
        stack = lambda key: torch.stack([s[key] for s in st])  # noqa: E731
        obs = {"state": stack("state"), "eligible": stack("eligible"), "epochs_mask": stack("epochs_mask")}
        if st[0]["window"] is not None:
            obs["window"] = stack("window")
        if st[0]["hidden"] is not None:
            obs["hidden"] = stack("hidden")
        act = {"selected": stack("selected")}
        for key in ACTION_KEYS:
            act[key] = stack(key)
        agents = {}
        for a in self.agent_names:
            adv = np.asarray(self._adv[a])
            norm = (adv - adv.mean()) / (adv.std() + 1e-8)
            T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
            agents[a] = {
                "logp": T([s["logp"][a] for s in st]), "value": T([s["value"][a] for s in st]),
                "adv": T(norm), "ret": T(self._ret[a]),
            }
        return {"obs": obs, "act": act, "agents": agents, "device": dev}

    def agent_full(self, agent: str) -> dict[str, torch.Tensor]:
        """All-transition tensors (old logp/value, normalized adv, returns) of one agent."""
        if self._cache is None:
            self._cache = self._build()
        return self._cache["agents"][agent]

    def batches(self, rng: np.random.Generator, n_minibatches: int | None = None) -> Iterator[dict]:
        """Yield random minibatches: {'idx','obs','act','agents':{name:{logp,value,adv,ret}}}."""
        if self._ep_start != len(self._steps):
            raise RuntimeError("RolloutBuffer has an unfinished episode; call finish_episode().")
        n = len(self._steps)
        if n == 0:
            return
        if self._cache is None:
            self._cache = self._build()
        c = self._cache
        nmb = max(1, min(n_minibatches or self.n_minibatches, n))
        for idx in np.array_split(rng.permutation(n), nmb):
            ti = torch.as_tensor(idx, dtype=torch.long, device=c["device"])
            yield {
                "idx": ti,
                "obs": {k: v[ti] for k, v in c["obs"].items()},
                "act": {k: v[ti] for k, v in c["act"].items()},
                "agents": {a: {k: v[ti] for k, v in d.items()} for a, d in c["agents"].items()},
            }
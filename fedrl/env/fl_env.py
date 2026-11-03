"""Gym-like FL environment (MASTER 2.5). One episode = T rounds of federated learning.

Works with any controller: ``act(obs, rng, deterministic) -> action dict``.
Drift for round t is applied (via FLWorld.apply_drift_if_due, which refreshes n and does
the free re-probe) while the observation preceding round t is built, so the policy always
sees post-drift data. This code never touches the official test split; evaluation-only code
reads env.world.G / env.world.L / env.scenario.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from fedrl.data.scenario import Scenario
from fedrl.data.scenario_sets import ScenarioSpec
from fedrl.env.memory import Memory
from fedrl.env.reward import compute_reward
from fedrl.env.state import build_state, state_dim
from fedrl.fl.fl_round import FLWorld
from fedrl.fl.resources import eligible_mask, epochs_mask
from fedrl.utils.logging_utils import get_logger
from fedrl.utils.seeding import make_rng

log = get_logger(__name__)

# action key -> (config menu name, config default-index key)
_MENUS = {
    "lr_idx": ("lr", "default_lr_idx"),
    "epochs_idx": ("epochs", "default_epochs_idx"),
    "batch_idx": ("batch", "default_batch_idx"),
    "alpha_idx": ("alpha", "default_alpha_idx"),
}


class FLEnv:
    """Federated-learning environment with per-client states and joint actions."""

    def __init__(self, cfg, bundle, pools, device) -> None:
        self.cfg, self.bundle, self.pools = cfg, bundle, pools
        self.device = torch.device(device)
        self.N = int(cfg.fl.N)
        self.k = int(cfg.fl.k)
        self.T = int(cfg.fl.T)
        self.state_dim = state_dim(cfg)
        self.memory = Memory(cfg, self.N, self.device)
        self.world: Optional[FLWorld] = None
        self.scenario: Optional[Scenario] = None
        self.spec: Optional[ScenarioSpec] = None
        self.t = 0
        self.drift_events = 0
        self.return_so_far = 0.0
        self.refs: Tuple[float, float] = (1.0, 1.0)
        self._fill_rng = make_rng(0, "env_fill")

    @property
    def n_probes(self) -> int:
        """Cumulative number of probes of the global model (FLWorld counter)."""
        return int(self.world.n_probes) if self.world is not None else 0

    # ------------------------------------------------------------------ reset
    def reset(self, spec: ScenarioSpec) -> Dict[str, Any]:
        """Start a new episode on ``spec`` and return the first observation."""
        cfg = self.cfg
        self.spec = spec
        self.scenario = Scenario(spec, self.bundle, self.pools, cfg)
        if int(self.scenario.N) != self.N:
            raise ValueError(f"scenario has N={self.scenario.N} clients but cfg.fl.N={self.N}")
        # FLWorld.__init__ calls reset(): fresh G, battery, counters, refs and the initial probe.
        self.world = FLWorld(self.scenario, self.bundle, self.pools, cfg, self.device)
        self.t, self.drift_events, self.return_so_far = 0, 0, 0.0
        self._fill_rng = make_rng(int(spec.seed), "env_fill")
        self.refs = (float(self.world.refs[0]), float(self.world.refs[1]))
        self.memory.reset()
        g_acc = float(self.world.global_val_acc()) if bool(cfg.state.extra_features) else 0.0
        self.memory.set_reference(g_acc, self.T * self.refs[0])
        self.memory.observe_probe(self.world.probe_loss, self.world.probe_acc)
        self._maybe_drift(1)
        return self._make_obs()

    # ------------------------------------------------------------------ helpers
    def _maybe_drift(self, round_t: int) -> None:
        """Apply drift due at ``round_t`` (1-based); FLWorld refreshes n and re-probes."""
        if not self.world.apply_drift_if_due(round_t):
            return
        self.drift_events += 1
        self.memory.observe_probe(self.world.probe_loss, self.world.probe_acc)

    def _make_obs(self) -> Dict[str, Any]:
        w, cfg, mem = self.world, self.cfg, self.memory
        props = torch.as_tensor(np.stack([c.proportions for c in self.scenario.clients]),
                                dtype=torch.float32, device=self.device)
        mode = str(cfg.state.memory_mode)
        state = build_state(
            cfg, proportions=props, n=w.n, probe_loss=mem.cur_loss, probe_acc=mem.cur_acc,
            prev_loss=mem.prev_loss, prev_acc=mem.prev_acc, last_div=mem.last_div,
            compute_speed=w.speed, bandwidth=w.bw, battery=w.battery,
            mem_features=mem.features(), round_idx=self.t,
            mem_delta=mem.delta_features() if mode == "ema_delta" else None,
            extras=mem.extra_features() if bool(cfg.state.extra_features) else None,
        )
        obs: Dict[str, Any] = {
            "state": state,
            "eligible": eligible_mask(w.battery, w.n, cfg).to(self.device),
            "epochs_mask": epochs_mask(w.battery, w.n, cfg).to(self.device),
            "round": int(self.t),
        }
        if mode == "gru_window":
            mem.push_state(state)
            obs["window"] = mem.window()
        return obs

    @staticmethod
    def _raw_selection(sel: Any) -> List[int]:
        if sel is None:
            return []
        return [int(v) for v in torch.as_tensor(sel).flatten().tolist() if int(v) >= 0]

    def _normalize_action(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Validate/repair an action: eligible, unique, length min(k, eligible), defaults filled."""
        cfg, w, N, dev = self.cfg, self.world, self.N, self.device
        elig = eligible_mask(w.battery, w.n, cfg).to(dev)
        emask = epochs_mask(w.battery, w.n, cfg).to(dev)
        k_eff = min(self.k, int(elig.sum().item()))
        raw = self._raw_selection(action.get("selected"))
        chosen: List[int] = []
        for i in raw:
            if 0 <= i < N and bool(elig[i]) and i not in chosen:
                chosen.append(i)
            if len(chosen) == k_eff:
                break
        repaired = chosen != raw
        if len(chosen) < k_eff:
            pool = [i for i in elig.nonzero().flatten().tolist() if i not in chosen]
            extra = self._fill_rng.choice(pool, size=k_eff - len(chosen), replace=False)
            chosen += [int(x) for x in extra]
        out: Dict[str, Any] = {"selected": torch.tensor(chosen, dtype=torch.long, device=dev)}
        for key, (menu, dflt) in _MENUS.items():
            size = len(getattr(cfg.menus, menu))
            if action.get(key) is None:
                t = torch.full((N,), int(getattr(cfg.menus, dflt)), dtype=torch.long, device=dev)
            else:
                t = torch.as_tensor(action[key], dtype=torch.long, device=dev).flatten().clone()
                if t.numel() != N:
                    raise ValueError(f"action[{key!r}] must have {N} entries, got {t.numel()}")
                t = t.clamp(0, size - 1)
            out[key] = t
        for i in chosen:  # masked epochs option -> largest feasible one
            if not bool(emask[i, out["epochs_idx"][i]]):
                feas = emask[i].nonzero().flatten()
                if feas.numel() == 0:
                    raise ValueError(f"client {i} has no feasible epochs option")
                out["epochs_idx"][i] = int(feas.max())
        for key in ("lr_value", "alpha_value"):
            if action.get(key) is not None:
                t = torch.as_tensor(action[key], dtype=torch.float32, device=dev).flatten().clone()
                if t.numel() != N:
                    raise ValueError(f"action[{key!r}] must have {N} entries, got {t.numel()}")
                out[key] = t
        if action.get("prox_mu") is not None:
            out["prox_mu"] = float(action["prox_mu"])
        if repaired:
            log.debug("selection repaired: raw=%s -> %s", raw, chosen)
        return out, repaired

    # ------------------------------------------------------------------ step
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Play one FL round. Returns (obs, reward, done, info)."""
        if self.world is None:
            raise RuntimeError("call reset(spec) before step()")
        if self.t >= self.T:
            raise RuntimeError("episode finished; call reset(spec)")
        cfg = self.cfg
        act, repaired = self._normalize_action(action)
        result = self.world.run_round(self.t + 1, act)   # includes the post-round probe
        self.t += 1
        done = self.t >= self.T
        reward, terms = compute_reward(result, self.refs, cfg, last_round=done)
        self.return_so_far += reward

        selected = [int(i) for i in result.selected]
        sel_t = torch.as_tensor(selected, dtype=torch.long, device=result.alpha_vals.device)
        self.memory.observe_probe(result.probe_loss, result.probe_acc)
        cos = None
        if bool(cfg.state.extra_features) and selected:
            cos = result.update_cos.detach().float().to(sel_t.device)[sel_t]
        self.memory.record_round(
            selected, result.divergence, update_cos=cos, rounds_since_local=result.rounds_since_local,
            global_acc=float(result.global_val_acc_after),
            global_acc_change=float(result.global_val_acc_after) - float(result.global_val_acc_before),
            comm=float(result.comm),
        )
        if not done:
            self._maybe_drift(self.t + 1)
        obs = self._make_obs()

        mean_alpha = float(result.alpha_vals.float()[sel_t].mean()) if selected else 0.0
        info: Dict[str, Any] = dict(terms)
        info.update({
            "reward": reward, "comm": float(result.comm), "comp": float(result.comp),
            "latency": float(result.latency), "mean_alpha": mean_alpha, "selected": selected,
            "epochs_vals": list(result.epochs_vals), "alpha_vals": result.alpha_vals,
            "pers_gain_chosen": float(result.pers_gain_chosen), "pers_gain_all": float(result.pers_gain_all),
            "global_val_acc": float(result.global_val_acc_after),
            "rounds_since_local": result.rounds_since_local, "n_probes": self.n_probes,
            "return_so_far": self.return_so_far, "round": self.t,
            "drift_events": self.drift_events, "selection_repaired": repaired,
        })
        return obs, float(reward), done, info
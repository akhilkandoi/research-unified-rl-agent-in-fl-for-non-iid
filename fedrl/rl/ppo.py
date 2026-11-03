"""PPO written from scratch: one PPOLearner per agent."""
from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import torch


class RunningStat:
    """Welford running mean/variance (used for optional reward normalization)."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: np.ndarray) -> None:
        for v in np.asarray(x, dtype=np.float64).ravel():
            self.count += 1
            d = v - self.mean
            self.mean += d / self.count
            self.m2 += d * (v - self.mean)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.m2 / self.count)) if self.count > 1 else 1.0

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, s: dict) -> None:
        self.count, self.mean, self.m2 = int(s["count"]), float(s["mean"]), float(s["m2"])


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    var = target.var(unbiased=False)
    if var.item() < 1e-12:
        return 0.0
    return float(1.0 - (target - pred).var(unbiased=False) / var)


class PPOLearner:
    """Clipped-objective PPO for one agent (own Adam optimizer, own reward normalizer)."""

    def __init__(self, params: Iterable[torch.nn.Parameter], cfg: Any, agent_name: str):
        self.agent_name = agent_name
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError(f"PPOLearner '{agent_name}' received no trainable parameters")
        p = cfg.ppo
        self.clip, self.epochs = float(p.clip), int(p.epochs)
        self.n_mb = int(p.minibatches)
        self.value_coef, self.entropy_coef = float(p.value_coef), float(p.entropy_coef)
        self.max_grad_norm = float(p.max_grad_norm)
        self.reward_norm = bool(p.reward_norm)
        self.opt = torch.optim.Adam(self.params, lr=float(p.lr))
        self.reward_stat = RunningStat()
        self.n_updates = 0

    def normalize_rewards(self, r: np.ndarray) -> np.ndarray:
        """Update running statistics and scale rewards by running std (no mean shift)."""
        self.reward_stat.update(r)
        return np.asarray(r) / (self.reward_stat.std + 1e-8) if self.reward_stat.count > 1 else np.asarray(r)

    def update(self, buffer: Any, evaluate_fn: Callable[[dict, dict], tuple],
               rng: np.random.Generator) -> dict[str, float]:
        """Run PPO epochs over the buffer. evaluate_fn(obs_b, act_b) -> (logp[B], entropy[B], value[B])
        for THIS agent. Returns mean stats."""
        name = self.agent_name
        full = buffer.agent_full(name)
        ev = explained_variance(full["value"], full["ret"])
        acc: dict[str, list[float]] = {k: [] for k in
                                       ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm")}
        for _ in range(self.epochs):
            for mb in buffer.batches(rng, self.n_mb):
                d = mb["agents"][name]
                logp, ent, val = evaluate_fn(mb["obs"], mb["act"])
                log_ratio = logp - d["logp"]
                ratio = log_ratio.exp()
                s1 = ratio * d["adv"]
                s2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * d["adv"]
                pl = -torch.min(s1, s2).mean()
                vl = ((val - d["ret"]) ** 2).mean()
                ent_m = ent.mean()
                loss = pl + self.value_coef * vl - self.entropy_coef * ent_m
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite PPO loss for agent '{name}': {loss.item()}")
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    acc["policy_loss"].append(pl.item())
                    acc["value_loss"].append(vl.item())
                    acc["entropy"].append(ent_m.item())
                    acc["approx_kl"].append(((ratio - 1.0) - log_ratio).mean().item())
                    acc["clip_frac"].append(((ratio - 1.0).abs() > self.clip).float().mean().item())
                    acc["grad_norm"].append(float(gn))
        self.n_updates += 1
        out = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
        out["explained_var"] = ev
        return out

    def state_dict(self) -> dict:
        return {"opt": self.opt.state_dict(), "reward_stat": self.reward_stat.state_dict(),
                "n_updates": self.n_updates}

    def load_state_dict(self, s: dict) -> None:
        self.opt.load_state_dict(s["opt"])
        self.reward_stat.load_state_dict(s["reward_stat"])
        self.n_updates = int(s["n_updates"])
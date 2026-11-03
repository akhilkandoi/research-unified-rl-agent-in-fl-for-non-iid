"""EXTENDED controllers (not part of the core experiments): ValueSelector, AdaptiveMixing."""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fedrl.controllers.base import (Controller, make_action, random_eligible_selection,
                                    top_k_eligible)
from fedrl.utils.flat import set_flat
from fedrl.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

VALUE_SELECTOR_DEFAULTS = {"eps": 0.1, "lr": 1e-3, "buffer": 5000, "batch": 64,
                           "target_sync": 50, "seed": 0}
ADAPTIVE_MIXING_DEFAULTS = {"lr": 1.0, "theta_clip": 4.0}


def _mlp(d_in: int, hidden: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class ValueSelector(Controller):
    """EXTENDED: shared MLP Q(state_i); epsilon-greedy top-k; TD(0) on chosen clients.

    Target for chosen client i: r + gamma * Q_target(next_state_i) (r if terminal), where
    r is the shared round reward. Settings are fixed; alpha is the menu default (0.5).
    """

    name = "value_selector"

    def __init__(self, cfg: Any, state_dim: int) -> None:
        super().__init__(cfg)
        p = {key: cfg.get(f"value_selector.{key}", val) for key, val in VALUE_SELECTOR_DEFAULTS.items()}
        self.p, self.dim, self.gamma = p, int(state_dim), float(cfg.ppo.gamma)
        hidden = list(cfg.get("policy.hidden", [64, 64]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(p["seed"]))
            self.net = _mlp(self.dim, hidden)
        self.target = copy.deepcopy(self.net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(p["lr"]))
        self.buffer: list[tuple] = []
        self.steps = 0
        self.rng = np.random.default_rng(int(p["seed"]))

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def act(self, obs: dict, rng: np.random.Generator, deterministic: bool = True) -> dict:
        if not deterministic and rng.random() < float(self.p["eps"]):
            sel = random_eligible_selection(obs, self.k, rng)
        else:
            with torch.no_grad():
                q = self._q(obs["state"].detach().cpu()).numpy()
            sel = top_k_eligible(obs, self.k, q)
        return make_action(sel, None, None, None, None, obs, self.cfg)

    def observe(self, obs: dict, action: dict, reward: float, next_obs: dict | None,
                info: dict) -> None:
        t_max = int(self.cfg.fl.T)
        done = bool(info.get("done", False)) or next_obs is None or \
            int(next_obs.get("round", 0)) >= t_max
        x = obs["state"].detach().cpu()
        xn = x if next_obs is None else next_obs["state"].detach().cpu()
        for i in action["selected"].cpu().tolist():
            self.buffer.append((x[i].clone(), float(reward), xn[i].clone(), float(done)))
        del self.buffer[: max(0, len(self.buffer) - int(self.p["buffer"]))]
        if len(self.buffer) >= int(self.p["batch"]):
            self._train_step()

    def _train_step(self) -> None:
        idx = self.rng.choice(len(self.buffer), size=int(self.p["batch"]), replace=False)
        x, r, xn, d = (torch.stack([self.buffer[i][j] for i in idx]) if j in (0, 2)
                       else torch.tensor([self.buffer[i][j] for i in idx]) for j in range(4))
        with torch.no_grad():
            y = r + self.gamma * (1.0 - d) * self.target(xn).squeeze(-1)
        loss = F.smooth_l1_loss(self._q(x), y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1
        if self.steps % int(self.p["target_sync"]) == 0:
            self.target.load_state_dict(self.net.state_dict())

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict(), "target": self.target.state_dict(),
                "opt": self.opt.state_dict(), "steps": self.steps,
                "buffer": [(a, b, c, d) for a, b, c, d in self.buffer],
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
        self.target.load_state_dict(state["target"])
        self.opt.load_state_dict(state["opt"])
        self.steps = int(state["steps"])
        self.buffer = [tuple(t) for t in state["buffer"]]
        self.rng.bit_generator.state = state["rng"]


class AdaptiveMixing(Controller):
    """EXTENDED non-RL: alpha_i = sigmoid(theta_i), one gradient step per round.

    After each round, for every chosen client, theta_i takes one SGD step on the
    client's personal-validation loss of B_i = alpha_i*G_new + (1-alpha_i)*L_i.
    Selection is random, settings fixed; alpha is passed through ``alpha_value``.
    Requires ``bind_env(env)`` (after_aggregation blend mode only).
    """

    name = "adaptive_mixing"

    def __init__(self, cfg: Any, state_dim: int | None = None) -> None:
        super().__init__(cfg)
        self.lr = float(cfg.get("adaptive_mixing.lr", ADAPTIVE_MIXING_DEFAULTS["lr"]))
        self.clip = float(cfg.get("adaptive_mixing.theta_clip", ADAPTIVE_MIXING_DEFAULTS["theta_clip"]))
        a0 = float(cfg.menus.alpha[int(cfg.menus.default_alpha_idx)])
        self.theta0 = math.log(a0 / (1.0 - a0))
        self.theta = torch.full((int(cfg.fl.N),), self.theta0, dtype=torch.float64)
        self._env: Any = None
        self._model: nn.Module | None = None
        self._data: tuple | None = None

    def bind_env(self, env: Any, data: tuple | None = None) -> None:
        if str(self.cfg.fl.blend_mode) != "after_aggregation":
            raise NotImplementedError("AdaptiveMixing supports blend_mode=after_aggregation only")
        self._env = env
        self._model = copy.deepcopy(env.world.model)
        if data is None:
            bundle = getattr(env, "bundle", None) or getattr(env.world, "bundle", None)
            if bundle is None:
                raise RuntimeError("AdaptiveMixing cannot find the dataset on env/env.world; "
                                   "call bind_env(env, data=(X, y))")
            data = (bundle.train_x, bundle.train_y)
        self._data = data

    def reset(self, obs: dict) -> None:
        self.theta = torch.full((int(obs["state"].shape[0]),), self.theta0, dtype=torch.float64)

    def alphas(self) -> torch.Tensor:
        return torch.sigmoid(self.theta)

    def act(self, obs: dict, rng: np.random.Generator, deterministic: bool = True) -> dict:
        sel = random_eligible_selection(obs, self.k, rng)
        return make_action(sel, None, None, None, None, obs, self.cfg,
                           alpha_value=self.alphas().float().to(obs["state"].device))

    def _grad_b(self, blended: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        assert self._model is not None and self._data is not None
        x_all, y_all = self._data
        idx = torch.as_tensor(idx, dtype=torch.long, device=x_all.device)
        set_flat(self._model, blended.detach())
        self._model.zero_grad(set_to_none=True)
        F.cross_entropy(self._model(x_all[idx]), y_all[idx]).backward()
        return torch.cat([p.grad.reshape(-1) for p in self._model.parameters()])

    def observe(self, obs: dict, action: dict, reward: float, next_obs: dict | None,
                info: dict) -> None:
        if self._env is None:
            raise RuntimeError("AdaptiveMixing needs controller.bind_env(env) before observe(); "
                               "without it alpha would silently never adapt")
        world, clients = self._env.world, self._env.scenario.clients
        g = world.G.detach()
        for i in action["selected"].cpu().tolist():
            local = world.L[i]
            if local is None:
                continue
            a = float(action["alpha_value"][i])
            blended = a * g + (1.0 - a) * local
            d_alpha = float(torch.dot(self._grad_b(blended, clients[i].personal_val_idx),
                                      (g - local).to(blended.device)))
            self.theta[i] = float(torch.clamp(self.theta[i] - self.lr * d_alpha * a * (1.0 - a),
                                              -self.clip, self.clip))


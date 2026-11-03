"""LinUCB contextual bandit over the client state (MASTER 2.12, D7)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fedrl.controllers.base import Controller, make_action, top_k_eligible


class LinUCB(Controller):
    """One shared linear model; top-k eligible by UCB; fixed settings, alpha 0.5.

    The shared round reward is credited to every chosen client. Greedy (c=0) when
    ``deterministic`` is True.
    """

    name = "bandit"

    def __init__(self, cfg: Any, state_dim: int) -> None:
        super().__init__(cfg)
        self.dim = int(state_dim)
        self.ridge = float(cfg.bandit.ridge)
        self.ucb_c = float(cfg.bandit.ucb_c)
        self.A = self.ridge * np.eye(self.dim)
        self.b = np.zeros(self.dim)
        self.n_updates = 0

    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def scores(self, x: np.ndarray, c: float) -> np.ndarray:
        """theta.x + c*sqrt(x^T A^-1 x) for every row of ``x``."""
        score = x @ self.theta()
        if c > 0:
            a_inv = np.linalg.inv(self.A)
            score = score + c * np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", x, a_inv, x), 0.0))
        return score

    def act(self, obs: dict, rng: np.random.Generator, deterministic: bool = True) -> dict:
        x = obs["state"].detach().cpu().numpy().astype(np.float64)
        sel = top_k_eligible(obs, self.k, self.scores(x, 0.0 if deterministic else self.ucb_c))
        return make_action(sel, None, None, None, None, obs, self.cfg)

    def observe(self, obs: dict, action: dict, reward: float, next_obs: dict | None,
                info: dict) -> None:
        x = obs["state"].detach().cpu().numpy().astype(np.float64)
        for i in action["selected"].cpu().tolist():
            self.A += np.outer(x[i], x[i])
            self.b += float(reward) * x[i]
            self.n_updates += 1

    def state_dict(self) -> dict:
        return {"A": torch.from_numpy(self.A.copy()), "b": torch.from_numpy(self.b.copy()),
                "n_updates": int(self.n_updates)}

    def load_state_dict(self, state: dict) -> None:
        a = np.asarray(state["A"], dtype=np.float64)
        b = np.asarray(state["b"], dtype=np.float64)
        if a.shape != (self.dim, self.dim) or b.shape != (self.dim,):
            raise ValueError(f"bandit state has shapes {a.shape}/{b.shape}, expected "
                             f"({self.dim},{self.dim})/({self.dim},)")
        self.A, self.b, self.n_updates = a.copy(), b.copy(), int(state["n_updates"])
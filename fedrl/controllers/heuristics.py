"""Loss-based client selection heuristic."""
from __future__ import annotations

from typing import Any

import numpy as np

from fedrl.controllers.base import (COMPUTE_SPEED_IDX, PROBE_LOSS_IDX, Controller, make_action,
                                    top_k_eligible)


class LossHeuristic(Controller):
    """Top-k eligible clients by probe loss; ties by compute speed; alpha 0.5."""

    name = "loss_heuristic"

    def act(self, obs: dict, rng: np.random.Generator, deterministic: bool = True) -> dict:
        state = obs["state"].detach().cpu().numpy()
        sel = top_k_eligible(obs, self.k, state[:, PROBE_LOSS_IDX], state[:, COMPUTE_SPEED_IDX])
        return make_action(sel, None, None, None, None, obs, self.cfg)
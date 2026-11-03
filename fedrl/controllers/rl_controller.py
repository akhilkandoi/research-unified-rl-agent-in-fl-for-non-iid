"""Controller wrapping a trained PPO policy (deterministic evaluation)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fedrl.controllers.base import Controller
from fedrl.utils.logging_utils import get_logger
from fedrl.utils.seeding import make_torch_gen

LOGGER = get_logger(__name__)
_POLICY_KEYS = ("policy", "policy_state", "policy_state_dict", "model")


def default_run_dir(cfg: Any, name: str, variant: str = "default", seed: int = 0) -> Path:
    from fedrl.methods import apply_method
    from fedrl.runs.run_manager import run_dir
    return Path(run_dir(apply_method(cfg, name), name, variant, seed))


class RLController(Controller):
    """Loads the BEST checkpoint of a run folder and acts with the trained policy."""

    def __init__(self, name: str, cfg: Any, state_dim: int, run_dir: Path | str | None = None,
                 seed: int = 0) -> None:
        from fedrl.config import resolve_device
        from fedrl.methods import METHODS, apply_method
        from fedrl.policy.joint_policy import JointPolicy
        from fedrl.policy.stacked import StackedPolicy
        from fedrl.runs.checkpoint import CheckpointManager

        cfg_m = apply_method(cfg, name)
        super().__init__(cfg_m)
        self.name = name
        self.device = resolve_device(cfg_m)
        run_path = Path(run_dir) if run_dir is not None else default_run_dir(cfg, name)
        ckpt = CheckpointManager(run_path).load_best() or CheckpointManager(run_path).load_latest()
        if ckpt is None:
            raise FileNotFoundError(f"no checkpoint found in {run_path} for method '{name}'; "
                                    f"train it first with scripts/train.py")
        kind = METHODS[name]["kind"]
        cls = JointPolicy if kind == "rl_joint" else StackedPolicy
        self.policy = cls(cfg_m, state_dim).to(self.device)
        self.policy.load_state_dict(self._policy_state(ckpt))
        self.policy.eval()
        self.seed = int(seed)
        self.gen = make_torch_gen(self.seed, "rl_controller", str(self.device))
        self.hidden: torch.Tensor | None = None
        self.run_path = run_path

    @staticmethod
    def _policy_state(ckpt: dict) -> dict:
        for key in _POLICY_KEYS:
            if key in ckpt:
                return ckpt[key]
        raise KeyError(f"checkpoint has none of the policy keys {_POLICY_KEYS}; "
                       f"keys: {sorted(ckpt)}")

    def reset(self, obs: dict) -> None:
        # Fresh seeded generator every episode -> identical eval for identical scenarios
        self.gen = make_torch_gen(self.seed, "rl_controller", str(self.device))
        self.hidden = self.policy.init_hidden(int(obs["state"].shape[0]))

    def act(self, obs: dict, rng: np.random.Generator, deterministic: bool = True) -> dict:
        if not deterministic:
            self.gen.manual_seed(int(rng.integers(0, 2**31 - 1)))
        o = dict(obs)
        if self.hidden is not None:
            o["hidden"] = self.hidden
        with torch.no_grad():
            out = self.policy.sample(o, self.gen, deterministic=deterministic)
        self.hidden = out.hidden  # gru_const: carried across rounds; None otherwise
        return out.action
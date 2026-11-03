"""Controller registry: build_controller covers every name in methods.all_method_names()."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fedrl.controllers.base import Controller
from fedrl.controllers.bandit import LinUCB
from fedrl.controllers.extended import AdaptiveMixing, ValueSelector
from fedrl.controllers.fixed import (FedAvg, FedProx, FixedAlpha, TunedFixedAlpha,
                                     TunedFixedSettings, load_tuning)
from fedrl.controllers.heuristics import LossHeuristic

__all__ = ["Controller", "build_controller", "load_controller_state"]


def load_controller_state(run_dir: Path | str | None) -> dict | None:
    """Learning state of a baseline controller from a run folder checkpoint, if any."""
    if run_dir is None or not Path(run_dir).exists():
        return None
    from fedrl.runs.checkpoint import CheckpointManager
    mgr = CheckpointManager(Path(run_dir))
    ckpt = mgr.load_best() or mgr.load_latest()
    if ckpt is None:
        return None
    return ckpt.get("controller", ckpt.get("controller_state"))

def _with_state(ctrl: Controller, run_dir: Path | str | None) -> Controller:
    state = load_controller_state(run_dir)
    if state:
        ctrl.load_state_dict(state)
    return ctrl


def _baseline_builders() -> dict[str, Callable[[Any, int, Any, str], Controller]]:
    return {
        "fedavg": lambda cfg, d, rd, n: FedAvg(cfg),
        "fixed_alpha_0.5": lambda cfg, d, rd, n: FixedAlpha(cfg, 0.5),
        "fixed_alpha_0.1": lambda cfg, d, rd, n: FixedAlpha(cfg, 0.1),
        "fedprox": lambda cfg, d, rd, n: FedProx(cfg),
        "loss_heuristic": lambda cfg, d, rd, n: LossHeuristic(cfg),
        "tuned_settings": lambda cfg, d, rd, n: TunedFixedSettings.from_tuning(
            cfg, load_tuning(cfg, n, rd)),
        "tuned_alpha": lambda cfg, d, rd, n: TunedFixedAlpha.from_tuning(
            cfg, load_tuning(cfg, n, rd)),
        "bandit": lambda cfg, d, rd, n: _with_state(LinUCB(cfg, d), rd),
        "value_selector": lambda cfg, d, rd, n: _with_state(ValueSelector(cfg, d), rd),
        "adaptive_mixing": lambda cfg, d, rd, n: AdaptiveMixing(cfg, d),
    }


def build_controller(name: str, cfg: Any, state_dim: int,
                     run_dir: Path | str | None = None) -> Controller:
    """Build the controller for method ``name``.

    RL names load the best checkpoint from ``run_dir``; tuned_* read ``tuning.json`` from
    ``run_dir`` (default: the method's run folder); bandit/value_selector resume their
    learned state from ``run_dir`` if a checkpoint exists.
    """
    from fedrl.methods import METHODS, apply_method, is_rl
    if name not in METHODS:
        raise KeyError(f"unknown method '{name}'; known: {sorted(METHODS)}")
    if is_rl(name):
        from fedrl.controllers.rl_controller import RLController
        return RLController(name, cfg, state_dim, run_dir)
    builders = _baseline_builders()
    if name not in builders:
        raise KeyError(f"method '{name}' has no controller builder; supported baselines: "
                       f"{sorted(builders)}")
    return builders[name](apply_method(cfg, name), state_dim, run_dir, name)
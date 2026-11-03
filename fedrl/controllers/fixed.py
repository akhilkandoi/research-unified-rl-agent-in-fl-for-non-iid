"""Fixed-setting baselines, validation-only tuning and best-baseline selection."""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from fedrl.controllers.base import (Controller, make_action, random_eligible_selection,
                                    rollout_episode)
from fedrl.utils.io import atomic_json
from fedrl.utils.logging_utils import get_logger
from fedrl.utils.seeding import make_rng

LOGGER = get_logger(__name__)

FEDPROX_DEFAULT_MU = 0.01
TUNING_VARIANT = "default"
TUNING_SEED = 0
NON_RL_BASELINES = ("fedavg", "fixed_alpha_0.5", "fixed_alpha_0.1", "fedprox",
                    "loss_heuristic", "tuned_settings", "tuned_alpha", "bandit")

Evaluator = Callable[[Controller, Any], dict]


def nearest_alpha_idx(cfg: Any, alpha: float) -> int:
    """Menu index of the alpha value closest to ``alpha``."""
    return int(np.argmin([abs(a - alpha) for a in cfg.menus.alpha]))


class FixedController(Controller):
    """Random eligible selection with constant settings (None = menu default)."""

    def __init__(self, cfg: Any, name: str, lr_idx: int | None = None,
                 epochs_idx: int | None = None, batch_idx: int | None = None,
                 alpha_idx: int | None = None, **overrides: Any) -> None:
        super().__init__(cfg)
        self.name = name
        self.lr_idx, self.epochs_idx = lr_idx, epochs_idx
        self.batch_idx, self.alpha_idx = batch_idx, alpha_idx
        self.overrides = overrides

    def act(self, obs: dict, rng: np.random.Generator, deterministic: bool = True) -> dict:
        sel = random_eligible_selection(obs, self.k, rng)
        return make_action(sel, self.lr_idx, self.epochs_idx, self.batch_idx, self.alpha_idx,
                           obs, self.cfg, **self.overrides)


class FedAvg(FixedController):
    """Random selection, default settings, alpha 1.0 (global model only; D6)."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg, "fedavg", alpha_value=1.0)


class FixedAlpha(FixedController):
    """Random selection, default settings, constant blend weight ``alpha``."""

    def __init__(self, cfg: Any, alpha: float) -> None:
        super().__init__(cfg, f"fixed_alpha_{alpha:g}", alpha_idx=nearest_alpha_idx(cfg, alpha),
                         alpha_value=float(alpha))
        self.alpha = float(alpha)


class FedProx(FixedController):
    """FedAvg plus proximal term (prox_mu 0.01) and alpha 1.0."""

    def __init__(self, cfg: Any) -> None:
        mu = float(cfg.fl.prox_mu) if float(cfg.fl.prox_mu) > 0 else FEDPROX_DEFAULT_MU
        super().__init__(cfg, "fedprox", alpha_value=1.0, prox_mu=mu)


class TunedFixedSettings(FixedController):
    """Best constant (lr, epochs, batch) from tuning; alpha menu default (0.5)."""

    def __init__(self, cfg: Any, lr_idx: int, epochs_idx: int, batch_idx: int) -> None:
        super().__init__(cfg, "tuned_settings", lr_idx=lr_idx, epochs_idx=epochs_idx,
                         batch_idx=batch_idx)

    @classmethod
    def from_tuning(cls, cfg: Any, payload: dict) -> "TunedFixedSettings":
        c = payload["chosen"]
        return cls(cfg, int(c["lr_idx"]), int(c["epochs_idx"]), int(c["batch_idx"]))


class TunedFixedAlpha(FixedController):
    """Best constant alpha menu index from tuning; default training settings."""

    def __init__(self, cfg: Any, alpha_idx: int) -> None:
        super().__init__(cfg, "tuned_alpha", alpha_idx=alpha_idx)

    @classmethod
    def from_tuning(cls, cfg: Any, payload: dict) -> "TunedFixedAlpha":
        return cls(cfg, int(payload["chosen"]["alpha_idx"]))


# ----------------------------------------------------------------------------- budget
class BudgetExceeded(RuntimeError):
    """Raised when tuning would use more FL rounds than the cap."""


class RoundBudget:
    """Counts FL rounds against a hard cap."""

    def __init__(self, cap: int) -> None:
        self.cap, self.used = int(cap), 0

    def check(self, n: int) -> None:
        if self.used + n > self.cap:
            raise BudgetExceeded(f"tuning needs {self.used + n} FL rounds but the cap is "
                                 f"{self.cap} (tuning.budget_rounds, or ppo.episodes*fl.T)")

    def charge(self, n: int) -> None:
        self.check(n)
        self.used += n


def tuning_budget(cfg: Any) -> int:
    """tuning.budget_rounds, or ppo.episodes * fl.T when null."""
    b = cfg.get("tuning.budget_rounds", None)
    return int(b) if b is not None else int(cfg.ppo.episodes) * int(cfg.fl.T)


# ----------------------------------------------------------------------------- paths
def _method_run_dir(cfg: Any, name: str) -> Path:
    from fedrl.methods import apply_method
    from fedrl.runs.run_manager import run_dir
    return Path(run_dir(apply_method(cfg, name), name, TUNING_VARIANT, TUNING_SEED))


def tuning_path(cfg: Any, name: str, run_dir: Path | str | None = None) -> Path:
    """Location of tuning.json for ``tuned_settings`` / ``tuned_alpha``."""
    base = Path(run_dir) if run_dir is not None else _method_run_dir(cfg, name)
    return base / "tuning.json"


def load_tuning(cfg: Any, name: str, run_dir: Path | str | None = None) -> dict:
    """Read tuning.json (given run folder first, then the default tuning folder)."""
    paths = []
    if run_dir is not None:
        paths.append(Path(run_dir) / "tuning.json")
    paths.append(tuning_path(cfg, name))
    for path in paths:
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"tuning.json not found (looked in {[str(p) for p in paths]}); "
                            f"run `python scripts/tune_baselines.py` first to tune '{name}'")


def validation_scores_path(cfg: Any) -> Path:
    from fedrl.runs.run_manager import run_dir
    return Path(run_dir(cfg, "_baselines", "validation", 0)) / "validation_scores.json"


def _spec_key(spec: Any) -> str:
    return f"a{spec.a:g}_{spec.drift}_s{spec.seed}"


def record_validation_score(cfg: Any, method: str, score: float, per_scenario: dict,
                            path: Path | None = None) -> Path:
    """Merge one method's validation score into validation_scores.json."""
    path = Path(path) if path is not None else validation_scores_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[method] = {"score": float(score), "per_scenario": per_scenario}
    atomic_json(data, path)
    return path


def select_best_baseline(cfg: Any, path: Path | None = None, allow_partial: bool = False) -> str:
    """Best non-RL baseline by mean VALIDATION return (never uses test results)."""
    path = Path(path) if path is not None else validation_scores_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run tune_baselines.py and "
                                f"evaluate_baselines_on_validation first")
    data = json.loads(path.read_text())
    missing = [m for m in NON_RL_BASELINES if m not in data]
    if missing and not allow_partial:
        raise ValueError(f"validation scores missing for {missing}; evaluate them first "
                         f"(or pass allow_partial=True)")
    scored = {m: data[m]["score"] for m in NON_RL_BASELINES if m in data}
    if not scored:
        raise ValueError("no baseline validation scores available")
    return max(scored, key=lambda m: (scored[m], -NON_RL_BASELINES.index(m)))


# ----------------------------------------------------------------------------- evaluation
def default_evaluator(cfg: Any) -> Evaluator:
    """Evaluator running a deterministic episode in a real FLEnv (loaded lazily, once)."""
    holder: dict = {}

    def _evaluate(controller: Controller, spec: Any) -> dict:
        if "env" not in holder:
            from fedrl.config import resolve_device
            from fedrl.data.datasets import load_dataset
            from fedrl.data.pools import build_pools
            from fedrl.env.fl_env import FLEnv
            device = resolve_device(cfg)
            bundle = load_dataset(cfg.run.dataset, cfg.run.data_dir, device)
            holder["bundle"] = bundle
            holder["env"] = FLEnv(cfg, bundle, build_pools(bundle, cfg), device)
        b = holder["bundle"]
        rng = make_rng(int(spec.seed), "tune_select")
        return rollout_episode(holder["env"], controller, spec, rng, deterministic=True,
                               data=(b.train_x, b.train_y))

    return _evaluate

def evaluate_on_validation(cfg: Any, controller: Controller, evaluator: Evaluator | None = None,
                           scenarios: Sequence | None = None) -> tuple[float, dict]:
    """Mean return of ``controller`` over the validation scenarios (and per scenario)."""
    from fedrl.data.scenario_sets import validation_scenarios
    scenarios = list(scenarios) if scenarios is not None else validation_scenarios(cfg)
    evaluator = evaluator or default_evaluator(cfg)
    per = {_spec_key(s): float(evaluator(controller, s)["return"]) for s in scenarios}
    return float(np.mean(list(per.values()))), per


def evaluate_baselines_on_validation(cfg: Any, state_dim: int | None = None,
                                     names: Sequence[str] | None = None,
                                     evaluator: Evaluator | None = None,
                                     path: Path | None = None) -> dict:
    """Score each non-RL baseline on validation scenarios and record the scores."""
    from fedrl.controllers import build_controller
    if state_dim is None:
        from fedrl.env.state import state_dim as _sd
        state_dim = _sd(cfg)
    scores = {}
    for name in names or NON_RL_BASELINES:
        score, per = evaluate_on_validation(cfg, build_controller(name, cfg, state_dim),
                                            evaluator)
        record_validation_score(cfg, name, score, per, path)
        scores[name] = score
        LOGGER.info("validation score %s = %.4f", name, score)
    return scores


# ----------------------------------------------------------------------------- tuning
def _tune(cfg: Any, method: str, candidates: list[dict], make_controller: Callable[[dict], Controller],
          evaluator: Evaluator | None, scenarios: Sequence | None,
          run_dir: Path | str | None, describe: Callable[[dict], dict]) -> dict:
    from fedrl.data.scenario_sets import validation_scenarios
    val = validation_scenarios(cfg)
    scenarios = list(scenarios) if scenarios is not None else list(val)
    outside = [s for s in scenarios if s not in set(val)]
    if outside:
        raise ValueError(f"tuning may only use validation scenarios; got {outside}")
    t_rounds = int(cfg.fl.T)
    budget = RoundBudget(tuning_budget(cfg))
    budget.check(len(candidates) * len(scenarios) * t_rounds)  # fail before running anything
    evaluator = evaluator or default_evaluator(cfg)

    results = []
    for cand in candidates:
        controller = make_controller(cand)
        per = {}
        for spec in scenarios:
            out = evaluator(controller, spec)
            budget.charge(int(out.get("rounds", t_rounds)))
            per[_spec_key(spec)] = float(out["return"])
        results.append({**cand, **describe(cand), "score": float(np.mean(list(per.values()))),
                        "per_scenario": per})
    best = max(range(len(results)), key=lambda i: results[i]["score"])  # first max wins
    from fedrl.config import config_hash
    payload = {"method": method, "chosen": {k: results[best][k] for k in candidates[best]},
               "chosen_values": describe(candidates[best]), "score": results[best]["score"],
               "candidates": results, "budget_cap": budget.cap, "rounds_used": budget.used,
               "val_scenarios": [asdict(s) for s in scenarios], "config_hash": config_hash(cfg)}
    path = tuning_path(cfg, method, run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, path)
    record_validation_score(cfg, method, payload["score"], results[best]["per_scenario"])
    LOGGER.info("%s: used %d of %d FL rounds; chosen=%s; wrote %s", method, budget.used,
                budget.cap, payload["chosen"], path)
    payload["path"] = str(path)
    return payload


def tune_settings(cfg: Any, evaluator: Evaluator | None = None,
                  scenarios: Sequence | None = None,
                  run_dir: Path | str | None = None) -> dict:
    """Search the 27 (lr, epochs, batch) combinations on validation scenarios only."""
    m = cfg.menus
    cands = [{"lr_idx": a, "epochs_idx": b, "batch_idx": c} for a, b, c in
             itertools.product(range(len(m.lr)), range(len(m.epochs)), range(len(m.batch)))]
    return _tune(cfg, "tuned_settings", cands,
                 lambda c: TunedFixedSettings(cfg, c["lr_idx"], c["epochs_idx"], c["batch_idx"]),
                 evaluator, scenarios, run_dir,
                 lambda c: {"lr": m.lr[c["lr_idx"]], "epochs": m.epochs[c["epochs_idx"]],
                            "batch": m.batch[c["batch_idx"]]})


def tune_alpha(cfg: Any, evaluator: Evaluator | None = None,
               scenarios: Sequence | None = None,
               run_dir: Path | str | None = None) -> dict:
    """Search the alpha menu on validation scenarios only."""
    m = cfg.menus
    cands = [{"alpha_idx": i} for i in range(len(m.alpha))]
    return _tune(cfg, "tuned_alpha", cands, lambda c: TunedFixedAlpha(cfg, c["alpha_idx"]),
                 evaluator, scenarios, run_dir, lambda c: {"alpha": m.alpha[c["alpha_idx"]]})
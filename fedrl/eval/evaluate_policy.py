"""Test-set evaluation of one controller on one scenario (MASTER 2.13) and validation helper.

Test reporting never uses validation-selection logic; evaluate_on_validation never touches test data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from fedrl.data.datasets import get_official_test
from fedrl.data.scenario_sets import ScenarioSpec, validation_scenarios
from fedrl.env.fl_env import FLEnv
from fedrl.eval import metrics as M
from fedrl.eval.metrics import result_path, validation_path  # re-exported
from fedrl.fl.evaluate import evaluate_vec
from fedrl.utils.flat import blend
from fedrl.utils.io import atomic_json
from fedrl.utils.logging_utils import get_logger
from fedrl.utils.seeding import make_rng

logger = get_logger(__name__)

REWARD_TERMS = ("r_global", "r_pers", "r_fair", "comm_norm", "comp_norm", "final_bonus")


def _as_list(x: Any) -> list[int]:
    return [int(v) for v in torch.as_tensor(x).flatten().tolist()]


def alpha_vector(action: dict, cfg: Any, n_clients: int, device: Any) -> torch.Tensor:
    """Per-client alpha of an action: alpha_value override if present, else menu[alpha_idx]."""
    if action.get("alpha_value") is not None:
        a = torch.as_tensor(action["alpha_value"], dtype=torch.float32, device=device).flatten()
        return a.expand(n_clients).clone() if a.numel() == 1 else a
    menu = torch.tensor(list(cfg.menus.alpha), dtype=torch.float32, device=device)
    return menu[torch.as_tensor(action["alpha_idx"]).to(device).long().flatten()]


def final_alpha(infos: list[dict], action: dict, cfg: Any, n_clients: int, device: Any) -> torch.Tensor:
    """Alpha of ALL clients in the last round (Part 4 info['alpha_vals']), else derived from the action."""
    a = infos[-1].get("alpha_vals")
    if a is not None:
        return torch.as_tensor(a, dtype=torch.float32, device=device).flatten()
    return alpha_vector(action, cfg, n_clients, device)

def personalized_vectors(world: Any, alpha: torch.Tensor) -> list[torch.Tensor]:
    """B_i = alpha_i*G + (1-alpha_i)*L_i; never-trained clients (L_i None) get G (D5)."""
    out = []
    for i in range(world.N):
        li = world.L[i]
        out.append(world.G.clone() if li is None else blend(float(alpha[i]), world.G, li))
    return out


def run_episode(controller: Any, env: FLEnv, spec: ScenarioSpec, eval_seed: int, cfg: Any,
                step_hook: Callable[[FLEnv, int], None] | None = None, data: Any = None) -> dict:
    """One deterministic episode following the Part 7 evaluation protocol."""
    obs = env.reset(spec)
    controller.seed = eval_seed
    controller.reset(obs)
    controller.bind_env(env, data=data)  # AFTER env.reset (it builds a new world)
    rng = make_rng(eval_seed, "eval_select")
    learn_mix = getattr(controller, "name", "") == "adaptive_mixing"  # only controller that needs observe()
    infos: list[dict] = []
    rewards: list[float] = []
    last_action: dict | None = None
    done, t = False, 0
    while not done:
        action = controller.act(obs, rng, deterministic=True)
        next_obs, reward, done, info = env.step(action)
        t += 1
        if learn_mix:
            controller.observe(obs, action, reward, next_obs, info)
        rewards.append(float(reward))
        infos.append(info)
        last_action = action
        if step_hook is not None:
            step_hook(env, t)
        obs = next_obs
        if t > env.T + 5:
            raise RuntimeError("environment did not terminate within T rounds")
    return {"infos": infos, "rewards": rewards, "last_action": last_action}

def collect_diagnostics(infos: list[dict], rewards: list[float], n_clients: int) -> dict:
    """Per-round diagnostics: alpha, selections, reward terms, personalization gains, staleness."""
    selected = [_as_list(i["selected"]) for i in infos]
    counts = np.zeros(n_clients, dtype=int)
    for s in selected:
        for c in s:
            counts[c] += 1
    rsl = [torch.as_tensor(i["rounds_since_local"]).float().cpu().numpy() for i in infos]
    return {
        "mean_alpha_per_round": [float(i["mean_alpha"]) for i in infos],
        "selected_per_round": selected,
        "selection_counts": counts.tolist(),
        "reward_per_round": [float(r) for r in rewards],
        "reward_terms": {k: [float(i.get(k, 0.0)) for i in infos] for k in M_TERMS()},
        "pers_gain_chosen_per_round": [float(i["pers_gain_chosen"]) for i in infos],
        "pers_gain_all_per_round": [float(i["pers_gain_all"]) for i in infos],
        "rounds_since_local_mean_per_round": [float(r.mean()) for r in rsl],
        "rounds_since_local_final": rsl[-1].tolist(),
        "comm_per_round": [float(i["comm"]) for i in infos],
        "comp_per_round": [float(i["comp"]) for i in infos],
        "latency_per_round": [float(i["latency"]) for i in infos],
    }


def M_TERMS() -> tuple[str, ...]:
    return REWARD_TERMS


def evaluate_controller(controller: Any, spec: ScenarioSpec, eval_seed: int, cfg: Any, bundle: Any,
                        pools: Any, device: Any, env: FLEnv | None = None) -> dict:
    """Run one TEST episode (deterministic controller) and compute all MASTER 2.13 metrics."""
    env = env if env is not None else FLEnv(cfg, bundle, pools, device)
    test_x, test_y = get_official_test(bundle)
    all_idx = torch.arange(test_y.shape[0], device=test_x.device)
    curve: list[float] = []

    def hook(e: FLEnv, t: int) -> None:  # evaluation-only per-round test accuracy
        curve.append(float(evaluate_vec(e.world.model, e.world.G, test_x, test_y, all_idx)[1]))

    ep = run_episode(controller, env, spec, eval_seed, cfg, step_hook=hook,
                     data=(bundle.train_x, bundle.train_y))
    world, infos = env.world, ep["infos"]
    n_clients = world.N
    alpha = final_alpha(infos, ep["last_action"], cfg, n_clients, world.G.device)
    vecs = personalized_vectors(world, alpha)
    pers_accs, glob_accs = [], []
    for i in range(n_clients):
        idx = env.scenario.personal_test_idx(i)
        pers_accs.append(float(evaluate_vec(world.model, vecs[i], test_x, test_y, idx)[1]))
        glob_accs.append(float(evaluate_vec(world.model, world.G, test_x, test_y, idx)[1]))
    ps = M.client_accuracy_stats(pers_accs, float(cfg.reward.fair_percentile))
    gs = M.client_accuracy_stats(glob_accs, float(cfg.reward.fair_percentile))
    dataset = str(cfg.run.dataset)
    target = cfg.get(f"eval.target_acc.{dataset}", None)
    win = M.drift_window(cfg, spec.drift)
    recovery = None if win is None else M.drift_recovery_time(curve, win[0], win[1])
    lat = [float(i["latency"]) for i in infos]
    metrics = {
        "global_test_acc": curve[-1],
        "pers_test_acc": ps["mean"],
        "worst10_acc": ps["worst"],
        "min_acc": ps["min"],
        "global_on_personal_acc": gs["mean"],
        "worst10_acc_global": gs["worst"],
        "comm": float(sum(i["comm"] for i in infos)),
        "comp": float(sum(i["comp"] for i in infos)),
        "latency": float(sum(lat)),
        "latency_mean": float(np.mean(lat)),
        "rounds_to_target": M.rounds_to_target(curve, target),
        "target_acc": target,
        "auc": M.area_under_curve(curve),
        "drift_recovery": recovery,
        "drift_start": None if win is None else win[0],
        "total_return": float(sum(ep["rewards"])),
    }
    result = {"metrics": metrics,
              "client_acc": {"personalized": pers_accs, "global": glob_accs,
                             "alpha_final": alpha.cpu().tolist()},
              "diagnostics": collect_diagnostics(infos, ep["rewards"], n_clients)}
    if bool(cfg.get("eval.log_curves", True)):
        result["curves"] = {"global_test_acc": curve,
                            "global_val_acc": [float(i["global_val_acc"]) for i in infos]}
    return result


def evaluate_and_save(controller: Any, spec: ScenarioSpec, eval_seed: int, cfg: Any, bundle: Any,
                      pools: Any, device: Any, run_dir: str | Path, meta: dict, tag: str = "",
                      force: bool = False, env: FLEnv | None = None) -> tuple[dict, bool]:
    """Idempotent (D16) evaluation: returns (result, computed_now). Existing JSON is reused."""
    path = result_path(run_dir, spec, eval_seed, tag)
    if path.exists() and not force:
        try:
            return json.loads(path.read_text()), False
        except json.JSONDecodeError:
            logger.warning("corrupt result %s, recomputing", path)
    res = evaluate_controller(controller, spec, eval_seed, cfg, bundle, pools, device, env=env)
    res["meta"] = {**meta, "dataset": str(cfg.run.dataset),
                   "scenario": {"a": spec.a, "drift": spec.drift, "seed": spec.seed},
                   "eval_seed": eval_seed, "tag": tag, "N": int(cfg.fl.N), "k": int(cfg.fl.k),
                   "T": int(cfg.fl.T), "schema": 1}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(M.to_jsonable(res), path)
    return res, True


def evaluate_on_validation(controller: Any, cfg: Any, bundle: Any, pools: Any, device: Any,
                           specs: list[ScenarioSpec] | None = None, env: FLEnv | None = None) -> dict:
    """Deterministic run on the VALIDATION scenarios; mean return and global/personalized val accuracy.

    Uses only train-set based validation indices; the official test set is never accessed here.
    """
    specs = list(specs) if specs is not None else validation_scenarios(cfg)
    env = env if env is not None else FLEnv(cfg, bundle, pools, device)
    rows = []
    for spec in specs:
        ep = run_episode(controller, env, spec, spec.seed, cfg, data=(bundle.train_x, bundle.train_y))
        world, infos = env.world, ep["infos"]
        alpha = final_alpha(infos, ep["last_action"], cfg, world.N, world.G.device)
        vecs = personalized_vectors(world, alpha)
        accs = []
        for i, c in enumerate(env.scenario.clients):
            idx = torch.as_tensor(c.personal_val_idx, device=world.G.device).long()
            accs.append(float(evaluate_vec(world.model, vecs[i], bundle.train_x, bundle.train_y, idx)[1]))
        rows.append({"a": spec.a, "drift": spec.drift, "seed": spec.seed,
                     "return": float(sum(ep["rewards"])),
                     "global_val_acc": float(infos[-1]["global_val_acc"]),
                     "pers_val_acc": float(np.mean(accs)),
                     "comm": float(sum(i["comm"] for i in infos)),
                     "comp": float(sum(i["comp"] for i in infos))})
    ret = np.array([r["return"] for r in rows])
    return {"mean_return": float(ret.mean()), "std_return": float(ret.std()),
            "global_val_acc": float(np.mean([r["global_val_acc"] for r in rows])),
            "pers_val_acc": float(np.mean([r["pers_val_acc"] for r in rows])),
            "n_scenarios": len(rows), "per_scenario": rows}


def save_validation_result(run_dir: str | Path, res: dict, extra: dict | None = None) -> Path:
    """Write validation.json (used for best-baseline selection and as the 'tune' done-marker)."""
    path = validation_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(M.to_jsonable({**res, **(extra or {})}), path)
    return path


def load_validation_result(run_dir: str | Path) -> dict | None:
    """Read validation.json if present."""
    p = validation_path(run_dir)
    return json.loads(p.read_text()) if p.exists() else None
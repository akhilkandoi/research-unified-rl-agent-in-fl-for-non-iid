"""Training loop: PPO for RL methods, observe-loop for learning controllers (e.g. bandit)."""
from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from fedrl.config import config_hash, resolve_device, validate
from fedrl.methods import METHODS, apply_method, apply_variant
from fedrl.rl.buffer import RolloutBuffer, step_rewards
from fedrl.rl.ppo import PPOLearner
from fedrl.runs.checkpoint import CheckpointManager
from fedrl.runs.locks import RunLock
from fedrl.runs.run_manager import ConfigMismatchError, prepare_run_dir
from fedrl.runs.train_log import TERMS, TrainLog
from fedrl.utils.logging_utils import get_logger
from fedrl.utils.seeding import (get_all_rng_states, make_rng, make_torch_gen,
                                 seed_everything, set_all_rng_states)

log = get_logger(__name__)


class TestAccessError(RuntimeError):
    """Training/checkpoint code asked for test scenarios or the official test set."""
    __test__ = False  # not a pytest class


@contextlib.contextmanager
def forbid_test_access():
    """While active, test-scenario / official-test accessors raise TestAccessError."""
    import fedrl.data.datasets as ds
    import fedrl.data.scenario_sets as ss
    targets = [(ss, "test_scenarios"), (ds, "get_official_test")]
    try:
        import fedrl.data.scenario as sc
        if hasattr(sc, "get_official_test"):
            targets.append((sc, "get_official_test"))
    except ImportError:
        pass
    saved = []

    def make(name: str) -> Callable:
        def _raise(*a, **k):
            raise TestAccessError(f"'{name}' must never be used during training/tuning/checkpoint selection.")
        return _raise

    for mod, name in targets:
        if hasattr(mod, name):
            saved.append((mod, name, getattr(mod, name)))
            setattr(mod, name, make(name))
    try:
        yield
    finally:
        for mod, name, orig in saved:
            setattr(mod, name, orig)


def check_schedule(cfg: Any) -> None:
    """checkpoint_every, validate_every and episodes must be multiples of update_every_episodes."""
    ue = int(cfg.ppo.update_every_episodes)
    if ue < 1:
        raise ValueError("ppo.update_every_episodes must be >= 1")
    for key in ("checkpoint_every", "validate_every", "episodes"):
        v = int(getattr(cfg.ppo, key))
        if v % ue != 0:
            raise ValueError(f"ppo.{key}={v} must be a multiple of ppo.update_every_episodes={ue}")


def agent_parameters(policy: Any, name: str):
    """Parameters belonging to one agent (joint: all parameters)."""
    if list(policy.agent_names) == ["joint"]:
        return list(policy.parameters())
    if hasattr(policy, "agent_parameters"):
        return list(policy.agent_parameters(name))
    if hasattr(policy, "agents"):
        return list(policy.agents[name].parameters())
    params = [p for n, p in policy.named_parameters() if n.startswith((f"{name}.", f"agents.{name}."))]
    if not params:
        raise AttributeError(f"Cannot find parameters of agent '{name}' in {type(policy).__name__}")
    return params


class _EpisodeStats:
    """Accumulates return, reward terms, alpha, entropy and selection counts."""

    def __init__(self, n_clients: int):
        self.ret = 0.0
        self.terms: dict[str, float] = defaultdict(float)
        self.alpha: list[float] = []
        self.ent: list[float] = []
        self.sel_ent: list[float] = []
        self.counts = np.zeros(n_clients, dtype=np.int64)
        self.steps = 0

    def add(self, r: float, info: dict, entropy: dict | None = None) -> None:
        self.ret += float(r)
        for t in TERMS:
            self.terms[t] += float(info.get(t, 0.0))
        if "mean_alpha" in info:
            self.alpha.append(float(info["mean_alpha"]))
        for s in info.get("selected", []):
            self.counts[int(s)] += 1
        if entropy:
            self.ent.append(sum(float(v) for v in entropy.values()))
            if "select" in entropy:
                self.sel_ent.append(float(entropy["select"]))
        self.steps += 1

    def as_dict(self) -> dict:
        nan = float("nan")
        out = {"return": self.ret, "mean_alpha": float(np.mean(self.alpha)) if self.alpha else nan,
               "policy_entropy": float(np.mean(self.ent)) if self.ent else nan,
               "sel_entropy": float(np.mean(self.sel_ent)) if self.sel_ent else
               (float(np.mean(self.ent)) if self.ent else nan),
               "steps": self.steps, "counts": self.counts}
        out.update({t: self.terms[t] for t in TERMS})
        return out


def run_rl_episode(env, policy, spec, gen, deterministic: bool, buffer: RolloutBuffer | None,
                   cfg: Any, agent_names: list[str]) -> dict:
    """One FL run with the PPO policy; stores transitions in buffer if given."""
    obs = env.reset(spec)
    hidden = policy.init_hidden(env.N)
    st = _EpisodeStats(env.N)
    done = False
    while not done:
        ob = dict(obs)
        if hidden is not None:
            ob["hidden"] = hidden
        with torch.no_grad():
            out = policy.sample(ob, gen, deterministic=deterministic)
        hidden = out.hidden
        nobs, r, done, info = env.step(out.action)
        if buffer is not None:
            buffer.add(ob, out, step_rewards(info, cfg, agent_names), done)
        st.add(r, info, out.entropy)
        obs = nobs
    return st.as_dict()


def run_controller_episode(env, ctrl, spec, rng, learn: bool) -> dict:
    """One FL run with a Controller; learning controllers get observe() calls."""
    obs = env.reset(spec)
    ctrl.reset(obs)
    st = _EpisodeStats(env.N)
    done = False
    while not done:
        action = ctrl.act(obs, rng, deterministic=not learn)
        nobs, r, done, info = env.step(action)
        if learn:
            ctrl.observe(obs, action, r, nobs, info)
        st.add(r, info, None)
        obs = nobs
    return st.as_dict()


def _eval_agent(policy, agent: str):
    def fn(obs_b, act_b):
        logp, ent, val = policy.evaluate(obs_b, act_b)
        return logp[agent], ent[agent], val[agent]
    return fn


def _better(new: float, rec: dict) -> bool:
    return rec.get("val_return") is None or new > rec["val_return"]


def train(cfg: Any, method: str, variant: str, seed: int,
          shard: tuple[int, int] | None = None) -> Path:
    """Train (or resume/extend) one run. cfg is the BASE config; method and variant are applied here."""
    if method not in METHODS:
        raise KeyError(f"Unknown method '{method}'")
    run_cfg = apply_variant(apply_method(cfg, method), variant, strict=False)
    validate(run_cfg)
    check_schedule(run_cfg)
    rd = prepare_run_dir(run_cfg, method, variant, seed, shard)
    with RunLock(rd / "train.lock"):
        with forbid_test_access():
            return _train_locked(run_cfg, method, variant, seed, rd)


def _train_locked(cfg: Any, method: str, variant: str, seed: int, rd: Path) -> Path:
    from fedrl.data.datasets import load_dataset
    from fedrl.data.pools import build_pools
    from fedrl.data.scenario_sets import training_scenario, validation_scenarios
    from fedrl.env.fl_env import FLEnv

    device = resolve_device(cfg)
    torch.set_num_threads(int(cfg.run.num_threads))
    seed_everything(seed)
    bundle = load_dataset(cfg.run.dataset, cfg.run.data_dir, device)
    pools = build_pools(bundle, cfg)
    env = FLEnv(cfg, bundle, pools, device)
    kind = METHODS[method]["kind"]

    policy = ctrl = None
    learners: dict[str, PPOLearner] = {}
    if kind == "rl_joint":
        from fedrl.policy.joint_policy import JointPolicy
        policy = JointPolicy(cfg, env.state_dim).to(device)
    elif kind == "rl_stacked":
        from fedrl.policy.stacked import StackedPolicy
        policy = StackedPolicy(cfg, env.state_dim).to(device)
    else:
        from fedrl.controllers import build_controller
        ctrl = build_controller(method, cfg, env.state_dim, rd)
        if not hasattr(ctrl, "state_dict"):
            log.info("Method '%s' has no learnable state; nothing to train (run dir %s).", method, rd)
            return rd
    if policy is not None:
        agent_names = list(policy.agent_names)
        learners = {a: PPOLearner(agent_parameters(policy, a), cfg, a) for a in agent_names}
    else:
        agent_names = ["controller"]

    policy_gen = make_torch_gen(seed, "policy", device)
    ppo_rng = make_rng(seed, "ppo_minibatch")
    ctrl_rng = make_rng(seed, "controller")
    buffer = RolloutBuffer(cfg, agent_names) if policy is not None else None
    tlog = TrainLog(rd, agent_names, env.N)
    mgr = CheckpointManager(rd)
    h = config_hash(cfg)

    def rng_state() -> dict:
        return get_all_rng_states({"policy_gen": policy_gen, "ppo_rng": ppo_rng, "ctrl_rng": ctrl_rng})

    def restore_rng(state: dict) -> None:
        set_all_rng_states(state, {"policy_gen": policy_gen, "ppo_rng": ppo_rng, "ctrl_rng": ctrl_rng})

    done_eps, updates, env_steps = 0, 0, 0
    best = {"val_return": None, "episode": -1}
    state = mgr.load_latest()
    if state is not None:
        if state.get("config_hash") != h:
            raise ConfigMismatchError(f"Checkpoint in {rd} has config hash {state.get('config_hash')} != {h}")
        done_eps, updates, env_steps = int(state["episode"]), int(state["updates"]), int(state["env_steps"])
        best = dict(state["best"])
        if policy is not None:
            policy.load_state_dict(state["policy"])
            for a, l in learners.items():
                l.load_state_dict(state["learners"][a])
        else:
            ctrl.load_state_dict(state["controller"])
        restore_rng(state["rng"])
        bst = mgr.load_best()
        if bst is not None and bst.get("best") and _better(bst["best"]["val_return"], best) \
                and bst["best"]["val_return"] is not None:
            best = dict(bst["best"])
        log.info("Resumed %s at episode %d (best val %s).", rd, done_eps, best["val_return"])
    tlog.truncate_after(done_eps)

    def run_episode(spec, train_mode: bool) -> dict:
        if ctrl is not None:
            rng = ctrl_rng if train_mode else make_rng(seed, "validation")
            return run_controller_episode(env, ctrl, spec, rng, learn=train_mode)
        gen = policy_gen if train_mode else make_torch_gen(seed, "validation", device)
        return run_rl_episode(env, policy, spec, gen, deterministic=not train_mode,
                              buffer=buffer if train_mode else None, cfg=cfg, agent_names=agent_names)

    def validate_now() -> float:
        return float(np.mean([run_episode(s, False)["return"] for s in validation_scenarios(cfg)]))

    def snapshot(E: int) -> dict:
        s = {"episode": E, "updates": updates, "env_steps": env_steps, "best": dict(best),
             "rng": rng_state(), "config": cfg.to_dict(), "config_hash": h,
             "method": method, "variant": variant, "seed": seed}
        if policy is not None:
            s["policy"] = policy.state_dict()
            s["learners"] = {a: l.state_dict() for a, l in learners.items()}
        else:
            s["controller"] = ctrl.state_dict()
        return s

    total = int(cfg.ppo.episodes)
    ue, ck, ve = (int(cfg.ppo.update_every_episodes), int(cfg.ppo.checkpoint_every),
                  int(cfg.ppo.validate_every))
    reward_fns = ({a: l.normalize_rewards for a, l in learners.items()}
                  if (learners and cfg.ppo.reward_norm) else None)
    pending: list[tuple[dict, np.ndarray]] = []
    try:
        for e in range(done_eps, total):
            t0 = time.time()
            spec = training_scenario(e, cfg)
            st = run_episode(spec, True)
            if buffer is not None:
                buffer.finish_episode(0.0, reward_fns)
            env_steps += st["steps"]
            row = {"episode": e + 1, "scenario_a": spec.a, "drift": spec.drift, "scenario_seed": spec.seed,
                   "return": st["return"], "mean_alpha": st["mean_alpha"],
                   "policy_entropy": st["policy_entropy"], "sel_entropy": st["sel_entropy"],
                   "env_steps": env_steps, "wall_time": time.time() - t0}
            row.update({t: st[t] for t in TERMS})
            pending.append((row, st["counts"]))
            log.info("ep %d/%d a=%.2f drift=%s return=%.3f", e + 1, total, spec.a, spec.drift, st["return"])
            E = e + 1
            if E % ue != 0:
                continue
            ppo_stats = None
            if policy is not None:
                ppo_stats = {a: l.update(buffer, _eval_agent(policy, a), ppo_rng) for a, l in learners.items()}
                buffer.clear()
                updates += 1
            for i, (r, counts) in enumerate(pending):
                tlog.log_episode(r, ppo_stats if i == len(pending) - 1 else None)
                tlog.log_selection(r["episode"], counts)
            pending = []
            if E % ve == 0:
                vr = validate_now()
                is_best = _better(vr, best)
                tlog.log_validation(E, vr, is_best)
                if is_best:
                    best = {"val_return": vr, "episode": E}
                    mgr.save_best(snapshot(E), E)
                log.info("validation at %d: return=%.3f best=%.3f", E, vr, best["val_return"])
            if E % ck == 0 or E == total:
                mgr.save(snapshot(E), E)
    except KeyboardInterrupt:
        log.warning("Interrupted; last checkpoint (episode %d) left intact. Re-run to resume.", mgr_last(mgr))
        raise
    log.info("Run complete: %s", rd)
    return rd


def mgr_last(mgr: CheckpointManager) -> int:
    ck = mgr.list_checkpoints()
    return ck[-1][0] if ck else 0
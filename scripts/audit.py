"""Static and dynamic audit of the repository. Prints PASS/FAIL per check, exit code 1 on failure."""
# ===== EDIT HERE =====
REPO_ROOT = None        # None = parent of scripts/; or a path string
SKIP = []               # check ids to skip, e.g. ["g"]
VERBOSE = True          # print details for passing checks too
# ===== END EDIT =====

import ast
import dataclasses
import importlib
import inspect
import math
import re
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(REPO_ROOT).resolve() if REPO_ROOT else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SELF = Path(__file__).resolve()

KEY_DEFAULTS = {  # the 30 most important defaults, MASTER section 4
    "fl.N": 20, "fl.k": 5, "fl.T": 30, "fl.blend_mode": "after_aggregation",
    "clients.size_mean": 600, "clients.size_sigma": 0.5, "clients.size_min": 150, "clients.size_max": 2000,
    "clients.min_battery": 0.05, "clients.recharge": 0.02, "data.pool_seed": 12345, "data.global_val_size": 5000,
    "menus.lr": [0.001, 0.01, 0.05], "menus.epochs": [1, 2, 5], "menus.batch": [16, 32, 64],
    "menus.alpha": [0.1, 0.3, 0.5, 0.7, 0.9], "state.memory_mode": "ema", "state.ema_beta": 0.9,
    "policy.hidden": [64, 64], "policy.heads": 4, "reward.acc_scale": 10.0, "reward.w_fair": 0.3,
    "reward.w_comm": 0.1, "reward.w_comp": 0.1, "reward.final_bonus": 1.0, "ppo.lr": 0.0003,
    "ppo.gamma": 0.97, "ppo.gae_lambda": 0.95, "ppo.clip": 0.2, "ppo.episodes": 400,
}
SMOKE_EXPECT = {"fl.N": 6, "fl.k": 2, "fl.T": 5, "drift.abrupt_round": 3, "drift.gradual_start": 3,
                "drift.gradual_rounds": 2, "ppo.episodes": 2, "ppo.update_every_episodes": 1,
                "ppo.checkpoint_every": 1, "ppo.validate_every": 2, "scenarios.val.seeds": [500],
                "scenarios.test.seeds": [1000], "clients.size_min": 40, "clients.size_max": 120,
                "clients.size_mean": 80, "data.global_val_size": 500, "ppo.epochs": 1, "ppo.minibatches": 1,
                "policy.hidden": [16, 16]}
PILOT_EXPECT = {"fl.N": 10, "fl.k": 3, "fl.T": 20, "drift.abrupt_round": 10, "drift.gradual_start": 10,
                "drift.gradual_rounds": 5, "ppo.episodes": 300, "policy.active_heads.train": False}
LAYOUT = """README.md pyproject.toml requirements.txt pytest.ini .gitignore ASSUMPTIONS.md
configs/default.yaml configs/smoke.yaml configs/pilot.yaml fedrl/__init__.py fedrl/config.py fedrl/methods.py
fedrl/experiments.py fedrl/utils/seeding.py fedrl/utils/flat.py fedrl/utils/io.py fedrl/utils/logging_utils.py
fedrl/models/cnn.py fedrl/data/datasets.py fedrl/data/pools.py fedrl/data/partition.py fedrl/data/drift.py
fedrl/data/scenario.py fedrl/data/scenario_sets.py fedrl/fl/resources.py fedrl/fl/costs.py fedrl/fl/client.py
fedrl/fl/server.py fedrl/fl/evaluate.py fedrl/fl/fl_round.py fedrl/env/state.py fedrl/env/memory.py
fedrl/env/reward.py fedrl/env/fl_env.py fedrl/policy/encoder.py fedrl/policy/attention.py
fedrl/policy/plackett_luce.py fedrl/policy/masking.py fedrl/policy/joint_policy.py fedrl/policy/stacked.py
fedrl/rl/buffer.py fedrl/rl/ppo.py fedrl/rl/trainer.py fedrl/runs/run_manager.py fedrl/runs/checkpoint.py
fedrl/runs/locks.py fedrl/runs/train_log.py fedrl/controllers/__init__.py fedrl/controllers/base.py
fedrl/controllers/fixed.py fedrl/controllers/heuristics.py fedrl/controllers/bandit.py
fedrl/controllers/rl_controller.py fedrl/controllers/extended.py fedrl/eval/metrics.py
fedrl/eval/evaluate_policy.py fedrl/eval/stats.py fedrl/eval/analysis.py scripts/train.py scripts/evaluate.py
scripts/run_all.py scripts/status.py scripts/analyze.py scripts/benchmark.py scripts/smoke_test.py
scripts/pilot.py scripts/tune_baselines.py scripts/reward_check.py scripts/audit.py""".split()

F = "fields"
CONTRACT = [
    ("fedrl.config", "load_config", ["path", "overrides"]), ("fedrl.config", "config_hash", ["cfg"]),
    ("fedrl.config", "validate", ["cfg"]), ("fedrl.config", "resolve_device", ["cfg"]),
    ("fedrl.utils.seeding", "make_rng", ["seed", "stream"]),
    ("fedrl.utils.seeding", "make_torch_gen", ["seed", "stream", "device"]),
    ("fedrl.utils.seeding", "seed_everything", ["seed"]), ("fedrl.utils.seeding", "get_all_rng_states", ["extra"]),
    ("fedrl.utils.seeding", "set_all_rng_states", ["state"]),
    ("fedrl.utils.flat", "get_flat", ["model"]), ("fedrl.utils.flat", "set_flat", ["model", "vec"]),
    ("fedrl.utils.flat", "num_params", ["model"]), ("fedrl.utils.flat", "weighted_average", ["vecs", "weights"]),
    ("fedrl.utils.flat", "blend", ["alpha", "g", "l"]), ("fedrl.utils.flat", "rel_div", ["w", "g"]),
    ("fedrl.utils.io", "atomic_write_bytes", ["path", "data"]), ("fedrl.utils.io", "atomic_torch_save", ["obj", "path"]),
    ("fedrl.utils.io", "atomic_json", ["obj", "path"]), ("fedrl.utils.logging_utils", "get_logger", ["name"]),
    ("fedrl.utils.logging_utils", "CsvLogger", ["path", "fields"]),
    ("fedrl.utils.logging_utils", "CsvLogger.append", ["row"]),
    ("fedrl.utils.logging_utils", "CsvLogger.truncate_after", ["episode"]),
    ("fedrl.models.cnn", "build_model", ["dataset"]),
    ("fedrl.data.datasets", "DatasetBundle", (F, ["name", "num_classes", "train_x", "train_y", "test_x", "test_y", "device"])),
    ("fedrl.data.datasets", "load_dataset", ["name", "data_dir", "device"]),
    ("fedrl.data.datasets", "get_official_test", ["bundle"]),
    ("fedrl.data.pools", "Pools", (F, ["global_val_idx", "train_by_class", "val_by_class"])),
    ("fedrl.data.pools", "build_pools", ["bundle", "cfg"]),
    ("fedrl.data.scenario_sets", "ScenarioSpec", (F, ["a", "drift", "seed"])),
    ("fedrl.data.scenario_sets", "training_scenario", ["episode", "cfg"]),
    ("fedrl.data.scenario_sets", "validation_scenarios", ["cfg"]), ("fedrl.data.scenario_sets", "test_scenarios", ["cfg"]),
    ("fedrl.data.scenario", "ClientData", (F, ["n", "proportions", "train_idx", "val_idx", "personal_val_idx",
                                                "compute_speed", "bandwidth", "battery0"])),
    ("fedrl.data.scenario", "Scenario", ["spec", "bundle", "pools", "cfg"]),
    ("fedrl.data.scenario", "Scenario.apply_drift", ["round_t"]),
    ("fedrl.data.scenario", "Scenario.personal_test_idx", ["i"]),
    ("fedrl.fl.resources", "drain", ["epochs", "n", "cfg"]), ("fedrl.fl.resources", "eligible_mask", ["battery", "n", "cfg"]),
    ("fedrl.fl.resources", "epochs_mask", ["battery", "n", "cfg"]),
    ("fedrl.fl.costs", "round_costs", ["chosen", "epochs_vals", "n", "speed", "bw", "model_bytes", "cfg"]),
    ("fedrl.fl.costs", "cost_reference", ["n", "speed", "bw", "model_bytes", "cfg"]),
    ("fedrl.fl.client", "local_train", ["model", "global_vec", "X", "y", "idx", "lr", "epochs", "batch", "prox_mu", "gen"]),
    ("fedrl.fl.server", "aggregate", ["vecs", "n_samples", "G", "mom_state", "m"]),
    ("fedrl.fl.evaluate", "evaluate_vec", ["model", "vec", "X", "y", "idx", "batch"]),
    ("fedrl.fl.evaluate", "per_class_acc", None),
    ("fedrl.fl.fl_round", "RoundResult", (F, ["selected", "epochs_vals", "global_val_acc_before", "global_val_acc_after",
                                               "fair_p10_before", "fair_p10_after", "pers_gain_chosen", "pers_gain_all",
                                               "comm", "comp", "latency", "divergence", "probe_loss", "probe_acc",
                                               "battery", "rounds_since_local", "times_picked", "idle_rounds",
                                               "alpha_vals"])),
    ("fedrl.fl.fl_round", "FLWorld", ["scenario", "bundle", "pools", "cfg", "device"]),
    ("fedrl.fl.fl_round", "FLWorld.reset", []), ("fedrl.fl.fl_round", "FLWorld.probe_all", []),
    ("fedrl.fl.fl_round", "FLWorld.run_round", ["round_t", "action"]), ("fedrl.fl.fl_round", "FLWorld.global_val_acc", []),
    ("fedrl.env.state", "state_dim", ["cfg"]), ("fedrl.env.state", "build_state", None),
    ("fedrl.env.memory", "Memory", ["cfg", "N"]),     ("fedrl.env.memory", "Memory.observe_probe", None), ("fedrl.env.memory", "Memory.record_round", None),
    ("fedrl.env.memory", "Memory.push_state", None),
    ("fedrl.env.memory", "Memory.features", []), ("fedrl.env.memory", "Memory.window", []),
    ("fedrl.env.memory", "Memory.reset", []),
    ("fedrl.env.reward", "compute_reward", ["result", "refs", "cfg", "last_round"]),
    ("fedrl.env.fl_env", "FLEnv", ["cfg", "bundle", "pools", "device"]),
    ("fedrl.env.fl_env", "FLEnv.reset", ["spec"]), ("fedrl.env.fl_env", "FLEnv.step", ["action"]),
    ("fedrl.policy.joint_policy", "PolicyOutput", (F, ["action", "logp", "entropy", "value", "hidden"])),
    ("fedrl.policy.joint_policy", "JointPolicy", ["cfg", "state_dim"]),
    ("fedrl.policy.joint_policy", "JointPolicy.sample", ["obs", "gen", "deterministic"]),
    ("fedrl.policy.joint_policy", "JointPolicy.evaluate", ["obs_b", "act_b"]),
    ("fedrl.policy.joint_policy", "JointPolicy.init_hidden", ["N"]),
    ("fedrl.policy.stacked", "StackedPolicy", ["cfg", "state_dim"]),
    ("fedrl.policy.stacked", "StackedPolicy.sample", ["obs", "gen", "deterministic"]),
    ("fedrl.policy.stacked", "StackedPolicy.evaluate", ["obs_b", "act_b"]),
    ("fedrl.policy.stacked", "StackedPolicy.init_hidden", ["N"]),
    ("fedrl.methods", "METHODS", None), ("fedrl.methods", "apply_method", ["cfg", "name"]),
    ("fedrl.methods", "is_rl", ["name"]), ("fedrl.methods", "all_method_names", []),
    ("fedrl.rl.ppo", "PPOLearner", ["params", "cfg", "agent_name"]), ("fedrl.rl.ppo", "PPOLearner.update", ["buffer", "evaluate_fn", "rng"]),
    ("fedrl.rl.ppo", "PPOLearner.state_dict", []), ("fedrl.rl.ppo", "PPOLearner.load_state_dict", None),
    ("fedrl.rl.buffer", "RolloutBuffer", ["cfg", "agent_names"]), ("fedrl.rl.buffer", "RolloutBuffer.add", None),
    ("fedrl.rl.buffer", "RolloutBuffer.finish_episode", ["last_value"]),
    ("fedrl.rl.buffer", "RolloutBuffer.batches", ["rng"]), ("fedrl.rl.buffer", "RolloutBuffer.clear", []),
    ("fedrl.rl.buffer", "RolloutBuffer.__len__", []),
    ("fedrl.rl.trainer", "train", ["cfg", "method", "variant", "seed", "shard"]),
    ("fedrl.runs.run_manager", "run_dir", ["cfg", "method", "variant", "seed"]),
    ("fedrl.runs.run_manager", "save_resolved_config", ["cfg", "path"]),
    ("fedrl.runs.checkpoint", "CheckpointManager", ["run_dir"]),
    ("fedrl.runs.checkpoint", "CheckpointManager.save", ["state", "episode", "is_best"]),
    ("fedrl.runs.checkpoint", "CheckpointManager.load_latest", []), ("fedrl.runs.checkpoint", "CheckpointManager.load_best", []),
    ("fedrl.controllers.base", "Controller", None), ("fedrl.controllers.base", "Controller.reset", ["obs"]),
    ("fedrl.controllers.base", "Controller.act", ["obs", "rng", "deterministic"]),
    ("fedrl.controllers.base", "Controller.observe", ["obs", "action", "reward", "next_obs", "info"]),
    ("fedrl.controllers", "build_controller", ["name", "cfg", "state_dim", "run_dir"]),
    ("fedrl.eval.evaluate_policy", "evaluate_controller", ["controller", "spec", "eval_seed", "cfg", "bundle", "pools", "device"]),
    ("fedrl.eval.stats", "paired_bootstrap", ["a", "b", "n_resamples", "ci", "rng"]),
    ("fedrl.experiments", "Job", (F, ["experiment", "dataset", "method", "variant", "train_seed", "scenario",
                                      "eval_seed", "kind"])),
    ("fedrl.experiments", "jobs_for", ["experiment", "cfg"]),
]


def py_files(*rel):
    base = ROOT.joinpath(*rel)
    return sorted(base.rglob("*.py")) if base.is_dir() else ([base] if base.exists() else [])


def rel(p):
    return str(p.relative_to(ROOT))


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def dig(d, dotted):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-9)
    return a == b


def _guard_ranges(tree):
    return [(n.lineno, n.end_lineno) for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "forbid_test_access"]


def _in(ranges, line):
    return any(a <= line <= b for a, b in ranges)

# ----------------------------------------------------------------------------- checks
def chk_a():
    bad = []
    for p in py_files("fedrl") + py_files("scripts"):
        if p.resolve() == SELF:
            continue
        text = p.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r"\b(TODO|FIXME)\b", line):
                bad.append(f"{rel(p)}:{i} TODO/FIXME")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Name) and node.id == "NotImplementedError":
                bad.append(f"{rel(p)}:{node.lineno} NotImplementedError")
            is_ell = lambda v: isinstance(v, ast.Constant) and v.value is Ellipsis  # noqa: E731
            if (isinstance(node, ast.Expr) and is_ell(node.value)) or \
               (isinstance(node, (ast.Assign, ast.Return)) and node.value is not None and is_ell(node.value)):
                bad.append(f"{rel(p)}:{node.lineno} ellipsis placeholder")
    return not bad, "; ".join(bad[:10]) or "none found"


def chk_b():
    allowed_files = {ROOT / "fedrl/data/scenario.py", ROOT / "fedrl/data/datasets.py"}
    bad = []
    for p in py_files("fedrl"):
        if (ROOT / "fedrl/eval") in p.parents or p in allowed_files:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        guard = _guard_ranges(tree)
        for node in ast.walk(tree):
            hit = ((isinstance(node, ast.ImportFrom) and any(a.name == "get_official_test" for a in node.names))
                   or (isinstance(node, ast.Attribute) and node.attr == "get_official_test")
                   or (isinstance(node, ast.Name) and node.id == "get_official_test"))
            if hit and not _in(guard, node.lineno):
                bad.append(f"{rel(p)}:{node.lineno}")
                break
    return not bad, "; ".join(bad) or "only fedrl/eval, scenario.py, scripts/, tests/ (trainer guard exempt)"

def chk_c():
    pat = re.compile(r"scenarios\s*(\.|\[\s*['\"])\s*test|test_scenarios\s*\(")
    files = (py_files("fedrl/rl") + py_files("fedrl/runs") + py_files("fedrl/env") +
             py_files("fedrl/controllers/fixed.py") + py_files("scripts/tune_baselines.py"))
    bad = []
    for p in files:
        text = p.read_text(encoding="utf-8")
        guard = _guard_ranges(ast.parse(text))
        bad += [f"{rel(p)}:{i}" for i, line in enumerate(text.splitlines(), 1)
                if pat.search(line) and not _in(guard, i)]
    return not bad, "; ".join(bad[:10]) or f"{len(files)} files clean"


def _attr_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def chk_d():
    default = load_yaml(ROOT / "configs/default.yaml")
    methods = {"get", "to_dict", "items", "keys", "values", "copy", "update", "set", "merge"}
    bad = set()
    for p in py_files("fedrl") + py_files("scripts"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                chain = _attr_chain(node)
                if not chain:
                    continue
                if chain[:2] == ["self", "cfg"]:
                    chain = chain[1:]
                if chain[0] not in ("cfg", "config") or len(chain) < 3 or chain[1] in methods:
                    continue
                cur = default
                for part in chain[1:]:
                    if not isinstance(cur, dict):
                        break
                    if part in methods:
                        break
                    if part not in cur:
                        bad.add(f"{rel(p)}: {'.'.join(chain)}")
                        break
                    cur = cur[part]
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                    and _attr_chain(node.func.value) in (["cfg"], ["config"], ["self", "cfg"]) \
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                try:
                    dig(default, node.args[0].value)
                except KeyError:
                    bad.add(f"{rel(p)}: get('{node.args[0].value}')")
    return not bad, "; ".join(sorted(bad)[:10]) or "all cfg keys exist in default.yaml"


def chk_e():
    default = load_yaml(ROOT / "configs/default.yaml")
    bad = []
    for key, want in KEY_DEFAULTS.items():
        try:
            got = dig(default, key)
        except KeyError:
            bad.append(f"{key} missing")
            continue
        if not same(got, want):
            bad.append(f"{key}: {got!r} != {want!r}")
    return not bad, "; ".join(bad) or f"{len(KEY_DEFAULTS)} keys match MASTER section 4"


def chk_f():
    from fedrl.models.cnn import build_model
    from fedrl.utils.flat import num_params
    model = build_model("mnist")
    n = sum(p.numel() for p in model.parameters())
    return n == 46730 and num_params(model) == 46730, f"{n} parameters"


def chk_g():
    from fedrl.config import load_config
    from fedrl.controllers import build_controller
    from fedrl.env.state import state_dim
    from fedrl.methods import METHODS, all_method_names, apply_method, is_rl
    from fedrl.policy.joint_policy import JointPolicy
    from fedrl.policy.stacked import StackedPolicy
    base = load_config(str(ROOT / "configs/smoke.yaml"), ["run.device=cpu"])
    bad, notes = [], []
    names = all_method_names()
    for name in names:
        try:
            kind = METHODS[name]["kind"]
            cfg = apply_method(base, name)
            sd = state_dim(cfg)
            if is_rl(name) != kind.startswith("rl"):
                bad.append(f"{name}: is_rl disagrees with kind {kind}")
            if kind == "rl_joint":
                JointPolicy(cfg, sd)
            elif kind == "rl_stacked":
                StackedPolicy(cfg, sd)
            elif kind == "baseline":
                try:
                    build_controller(name, cfg, sd, None)
                except Exception:  # noqa: BLE001
                    if name.startswith("tuned_"):
                        notes.append(name)
                    else:
                        raise
            else:
                bad.append(f"{name}: unknown kind {kind}")
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{name}: {type(exc).__name__}: {exc}")
    extra = f" (need tuning files: {', '.join(notes)})" if notes else ""
    return not bad, ("; ".join(bad) or f"{len(names)} methods build") + extra


def chk_h():
    from fedrl.config import config_hash, load_config
    smoke = str(ROOT / "configs/smoke.yaml")
    h = config_hash(load_config(smoke))
    ok = (h == config_hash(load_config(smoke, ["ppo.episodes=999"])) == config_hash(load_config(smoke, ["logging.level=DEBUG"]))
          and h != config_hash(load_config(smoke, ["fl.k=3"])) and len(h) == 12)
    return ok, f"hash {h}"


def chk_i():
    bad = []
    for p in py_files("fedrl"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        ok = [(n.lineno, n.end_lineno) for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name.startswith("print_")]  # e.g. print_parameter_report
        bad += [f"{rel(p)}:{n.lineno}" for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
                and not _in(ok, n.lineno)]
    return not bad, "; ".join(bad[:10]) or "no print() in fedrl/ (except print_* helpers)"

def chk_j():
    problems = []
    for mod, qual, exp in CONTRACT:
        try:
            obj = importlib.import_module(mod)
            for part in qual.split("."):
                obj = getattr(obj, part)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{mod}.{qual}: {type(exc).__name__}: {exc}")
            continue
        if exp is None:
            continue
        if isinstance(exp, tuple):
            names = [f.name for f in dataclasses.fields(obj)] if dataclasses.is_dataclass(obj) else []
            miss = [n for n in exp[1] if n not in names]
            if miss:
                problems.append(f"{mod}.{qual}: missing fields {miss}")
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            problems.append(f"{mod}.{qual}: no signature")
            continue
        names = [n for n, p in sig.parameters.items() if n not in ("self", "cls")
                 and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
        if names[:len(exp)] != exp:
            problems.append(f"{mod}.{qual}: expected leading params {exp}, found {names}")
    return not problems, "; ".join(problems[:12]) or f"{len(CONTRACT)} contract items match"


def chk_k():
    missing = [f for f in LAYOUT if not (ROOT / f).exists()]
    return not missing, ", ".join(missing[:15]) or f"{len(LAYOUT)} files present"


def chk_l():
    bad = []
    for p in py_files("scripts"):
        text = p.read_text(encoding="utf-8")
        if "# ===== EDIT HERE =====" not in text or "# ===== END EDIT =====" not in text:
            bad.append(f"{rel(p)}: no EDIT HERE block")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.keyword) and node.arg == "required" and isinstance(node.value, ast.Constant) \
                    and node.value.value is True:
                bad.append(f"{rel(p)}:{node.value.lineno} required=True argument")
    return not bad, "; ".join(bad[:10]) or "all scripts runnable without flags"


def chk_m():
    default = load_yaml(ROOT / "configs/default.yaml")
    bad = []
    for fname, expect in (("smoke", SMOKE_EXPECT), ("pilot", PILOT_EXPECT)):
        merged = deep_merge(default, load_yaml(ROOT / f"configs/{fname}.yaml") or {})
        for key, want in expect.items():
            try:
                got = dig(merged, key)
            except KeyError:
                bad.append(f"{fname}:{key} missing")
                continue
            if not same(got, want):
                bad.append(f"{fname}:{key}={got!r} want {want!r}")
    return not bad, "; ".join(bad) or "smoke.yaml and pilot.yaml match D13"


CHECKS = [("a", "no TODO/FIXME/ellipsis/NotImplementedError", chk_a), ("b", "get_official_test import scope", chk_b),
          ("c", "no test scenarios in training-side code", chk_c), ("d", "cfg keys exist in default.yaml", chk_d),
          ("e", "30 key defaults equal MASTER section 4", chk_e), ("f", "MNIST model has 46,730 parameters", chk_f),
          ("g", "every method builds", chk_g), ("h", "config hash ignores ppo.episodes", chk_h),
          ("i", "no print() in library modules", chk_i), ("j", "contract signatures", chk_j),
          ("k", "repository layout complete", chk_k), ("l", "scripts need no CLI arguments", chk_l),
          ("m", "smoke/pilot configs match D13", chk_m)]


def main() -> int:
    failures = 0
    for cid, title, fn in CHECKS:
        if cid in SKIP:
            print(f"SKIP ({cid}) {title}")
            continue
        try:
            ok, detail = fn()
        except Exception:  # noqa: BLE001
            ok, detail = False, traceback.format_exc(limit=4)
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} ({cid}) {title}" + (f" - {detail}" if (VERBOSE or not ok) else ""))
    print(f"{len(CHECKS) - failures - len(SKIP)} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
"""Configuration loading, hashing, validation and device resolution."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

MEMORY_MODES = ("none", "ema", "ema_delta", "gru_window", "gru_const")
BLEND_MODES = ("after_aggregation", "before_training")
REWARD_MODES = ("joint", "stacked_shared", "stacked_specialized")
DATASETS = ("mnist", "fmnist", "cifar10")
DRIFTS = ("none", "abrupt", "gradual")

# Keys excluded from the config hash (do not affect results / allow extending runs).
HASH_EXCLUDE = ("run", "ppo.episodes", "logging", "tuning.budget_rounds", "eval.log_curves")

_SCI = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


class ConfigError(ValueError):
    """Raised for invalid configuration content."""


class Config:
    """Dict wrapper with attribute access. Sub-configs are live views of the same data."""

    __slots__ = ("_d",)

    def __init__(self, data: dict | None = None) -> None:
        object.__setattr__(self, "_d", data if data is not None else {})

    # -- access -----------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        d = object.__getattribute__(self, "_d")
        if name not in d:
            raise AttributeError(f"Config has no key '{name}'. Available: {sorted(d)}")
        v = d[name]
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, name: str, value: Any) -> None:
        self._d[name] = value.to_dict() if isinstance(value, Config) else value

    def __getitem__(self, name: str) -> Any:
        return self.__getattr__(name) if name in self._d else self._raise_key(name)

    def _raise_key(self, name: str):
        raise KeyError(f"Config has no key '{name}'. Available: {sorted(self._d)}")

    def __setitem__(self, name: str, value: Any) -> None:
        self.__setattr__(name, value)

    def __contains__(self, name: object) -> bool:
        return name in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def keys(self):
        return self._d.keys()

    def items(self):
        return [(k, Config(v) if isinstance(v, dict) else v) for k, v in self._d.items()]

    def get(self, path: str, default: Any = None) -> Any:
        """Dotted-path lookup: cfg.get("ppo.lr", 3e-4). Returns default if any part is missing."""
        cur: Any = self._d
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return Config(cur) if isinstance(cur, dict) else cur

    def set(self, path: str, value: Any) -> None:
        """Dotted-path assignment; creates intermediate dicts."""
        parts = path.split(".")
        cur = self._d
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = value.to_dict() if isinstance(value, Config) else value

    # -- conversion / copying ---------------------------------------------------
    def to_dict(self) -> dict:
        """Deep copy as a plain dict."""
        return copy.deepcopy(self._d)

    def __deepcopy__(self, memo: dict) -> "Config":
        return Config(copy.deepcopy(self._d, memo))

    def __reduce__(self):
        return (Config, (self.to_dict(),))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            return self._d == other._d
        if isinstance(other, dict):
            return self._d == other
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Config({json.dumps(self._d, sort_keys=True, default=str)})"


# ----------------------------------------------------------------------------- parsing
def _fix_sci(v: Any) -> Any:
    """Convert strings like '1e-3' (read as str by PyYAML) to floats, recursively."""
    if isinstance(v, str) and _SCI.match(v.strip()):
        return float(v)
    if isinstance(v, list):
        return [_fix_sci(x) for x in v]
    if isinstance(v, dict):
        return {k: _fix_sci(x) for k, x in v.items()}
    return v


def parse_value(text: str) -> Any:
    """Typed parsing of an override value: int, float, bool, null, list, dict, else string."""
    try:
        v = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    return _fix_sci(v)


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at top level.")
    return _fix_sci(data)


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    for cand in (p, REPO_ROOT / p, REPO_ROOT / "configs" / p.name):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Config file not found: {path}")


def deep_merge(base: dict, over: dict, strict: bool = True, _prefix: str = "") -> dict:
    """Merge `over` into `base` in place. Lists/scalars replace; dicts merge recursively."""
    for key, val in over.items():
        full = f"{_prefix}{key}"
        if key not in base:
            if strict:
                raise ConfigError(f"Unknown config key '{full}' (not present in default.yaml).")
            base[key] = copy.deepcopy(val)
        elif isinstance(base[key], dict) and isinstance(val, dict):
            deep_merge(base[key], val, strict, full + ".")
        else:
            base[key] = copy.deepcopy(val)
    return base


def _apply_override(base: dict, item: str) -> None:
    if "=" not in item:
        raise ConfigError(f"Override '{item}' must look like 'a.b.c=value'.")
    key, text = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"Override '{item}' has an empty key.")
    nested: Any = parse_value(text.strip())
    for part in reversed(key.split(".")):
        nested = {part: nested}
    deep_merge(base, nested, strict=True)


def load_config(
    path: str | None = None,
    overrides: list[str] | None = None,
    validate_cfg: bool = True,
) -> Config:
    """Load default.yaml, then `path` (if given), then 'a.b=value' overrides; optionally validate."""
    base = _read_yaml(DEFAULT_CONFIG_PATH)
    if path:
        p = _resolve_path(path)
        if p.resolve() != DEFAULT_CONFIG_PATH.resolve():
            deep_merge(base, _read_yaml(p), strict=True)
    for item in overrides or []:
        _apply_override(base, item)
    cfg = Config(base)
    if validate_cfg:
        validate(cfg)
    return cfg


# ----------------------------------------------------------------------------- hashing
def _drop_path(d: dict, path: str) -> None:
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.get(part)
        if not isinstance(cur, dict):
            return
    cur.pop(parts[-1], None)


def _canon(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def config_hash(cfg: Config) -> str:
    """12 hex chars of sha256 over canonical JSON of result-affecting keys only."""
    d = cfg.to_dict() if isinstance(cfg, Config) else copy.deepcopy(cfg)
    for path in HASH_EXCLUDE:
        _drop_path(d, path)
    blob = json.dumps(_canon(d), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------------------- validation
def validate(cfg: Config) -> None:
    """Raise ConfigError listing every problem found."""
    errs: list[str] = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            errs.append(msg)

    def g(path: str, default: Any = None) -> Any:
        return cfg.get(path, default)

    N, k, T = g("fl.N"), g("fl.k"), g("fl.T")
    need(isinstance(N, int) and N >= 1, f"fl.N must be a positive int, got {N!r}.")
    need(isinstance(T, int) and T >= 1, f"fl.T must be a positive int, got {T!r}.")
    need(isinstance(k, int) and k >= 1, f"fl.k must be a positive int, got {k!r}.")
    if isinstance(N, int) and isinstance(k, int):
        need(k <= N, f"fl.k ({k}) must be <= fl.N ({N}).")
    need(g("fl.blend_mode") in BLEND_MODES, f"fl.blend_mode must be one of {BLEND_MODES}, got {g('fl.blend_mode')!r}.")
    need(g("run.dataset") in DATASETS, f"run.dataset must be one of {DATASETS}, got {g('run.dataset')!r}.")
    dev = str(g("run.device", "auto"))
    need(re.match(r"^(auto|cpu|cuda(:\d+)?|mps)$", dev) is not None, f"run.device invalid: {dev!r}.")
    nt = g("run.num_threads", 1)
    need(isinstance(nt, int) and nt >= 1, f"run.num_threads must be an int >= 1, got {nt!r}.")
    need(0.0 <= float(g("fl.server_momentum", 0.0)) < 1.0, "fl.server_momentum must be in [0, 1).")
    need(float(g("fl.prox_mu", 0.0)) >= 0.0, "fl.prox_mu must be >= 0.")

    # menus
    for name in ("lr", "epochs", "batch", "alpha"):
        m = g(f"menus.{name}")
        ok = isinstance(m, list) and len(m) > 0
        need(ok, f"menus.{name} must be a non-empty list.")
        if ok:
            idx = g(f"menus.default_{name}_idx")
            need(isinstance(idx, int) and 0 <= idx < len(m),
                 f"menus.default_{name}_idx={idx!r} out of range for menus.{name} (len {len(m)}).")
    for name in ("lr",):
        m = g(f"menus.{name}", [])
        need(all(isinstance(x, (int, float)) and x > 0 for x in m), "menus.lr values must be > 0.")
    for name in ("epochs", "batch"):
        m = g(f"menus.{name}", [])
        need(all(isinstance(x, int) and x >= 1 for x in m), f"menus.{name} values must be ints >= 1.")
    am = g("menus.alpha", [])
    need(all(isinstance(x, (int, float)) and 0.0 < x < 1.0 for x in am),
         f"menus.alpha values must lie strictly inside (0, 1), got {am!r}.")

    # state / policy / reward
    mm = g("state.memory_mode")
    need(mm in MEMORY_MODES, f"state.memory_mode must be one of {MEMORY_MODES}, got {mm!r}.")
    need(0.0 <= float(g("state.ema_beta", 0.9)) < 1.0, "state.ema_beta must be in [0, 1).")
    need(int(g("state.gru_window", 5)) >= 1, "state.gru_window must be >= 1.")
    for key in ("loss_scale", "div_scale", "log_clip"):
        need(float(g(f"state.{key}", 1.0)) > 0, f"state.{key} must be > 0.")
    need(g("reward.mode") in REWARD_MODES, f"reward.mode must be one of {REWARD_MODES}, got {g('reward.mode')!r}.")
    need(0 < float(g("reward.fair_percentile", 10)) < 100, "reward.fair_percentile must be in (0, 100).")
    hid = g("policy.hidden")
    need(isinstance(hid, list) and len(hid) >= 1 and all(isinstance(x, int) and x >= 1 for x in hid),
         f"policy.hidden must be a non-empty list of positive ints, got {hid!r}.")
    heads = g("policy.heads", 4)
    need(isinstance(heads, int) and heads >= 1, "policy.heads must be a positive int.")
    if isinstance(hid, list) and hid and isinstance(hid[-1], int) and isinstance(heads, int) and heads >= 1:
        need(hid[-1] % heads == 0,
             f"last policy.hidden size ({hid[-1]}) must be divisible by policy.heads ({heads}).")
    ah = g("policy.active_heads")
    need(isinstance(ah, Config) and all(h in ah and isinstance(ah[h], bool) for h in ("select", "train", "mix")),
         "policy.active_heads must define boolean select, train and mix.")

    # ppo
    upd = g("ppo.update_every_episodes")
    need(isinstance(upd, int) and upd >= 1, f"ppo.update_every_episodes must be an int >= 1, got {upd!r}.")
    if isinstance(upd, int) and upd >= 1:
        for key in ("checkpoint_every", "validate_every"):
            v = g(f"ppo.{key}")
            need(isinstance(v, int) and v >= 1 and v % upd == 0,
                 f"ppo.{key} ({v!r}) must be a positive multiple of ppo.update_every_episodes ({upd}).")
    need(isinstance(g("ppo.episodes"), int) and g("ppo.episodes") >= 1, "ppo.episodes must be an int >= 1.")
    need(int(g("ppo.epochs", 1)) >= 1 and int(g("ppo.minibatches", 1)) >= 1,
         "ppo.epochs and ppo.minibatches must be >= 1.")
    need(0 < float(g("ppo.clip", 0.2)) < 1, "ppo.clip must be in (0, 1).")

    # clients / data / drift
    smin, smax, smean = g("clients.size_min"), g("clients.size_max"), g("clients.size_mean")
    need(all(isinstance(x, (int, float)) for x in (smin, smax, smean)), "clients.size_* must be numbers.")
    if all(isinstance(x, (int, float)) for x in (smin, smax, smean)):
        need(0 < smin <= smax, f"clients.size_min ({smin}) must be in (0, size_max ({smax})].")
        need(smin <= smean <= smax, f"clients.size_mean ({smean}) must lie in [size_min, size_max].")
    need(float(g("clients.min_battery", 0.05)) >= 0, "clients.min_battery must be >= 0.")
    need(float(g("clients.battery_init_low", 0.5)) <= float(g("clients.battery_init_high", 1.0)),
         "clients.battery_init_low must be <= battery_init_high.")
    need(float(g("data.global_val_size", 1)) >= 1, "data.global_val_size must be >= 1.")
    need(0 < float(g("data.client_train_frac", 0.8)) < 1, "data.client_train_frac must be in (0, 1).")
    need(0 <= float(g("data.personal_val_own_frac", 0.8)) <= 1, "data.personal_val_own_frac must be in [0, 1].")
    if isinstance(T, int):
        ar, gs, gr = g("drift.abrupt_round"), g("drift.gradual_start"), g("drift.gradual_rounds")
        need(isinstance(ar, int) and 1 <= ar <= T, f"drift.abrupt_round ({ar!r}) must be in [1, fl.T={T}].")
        need(isinstance(gs, int) and isinstance(gr, int) and gr >= 1 and 1 <= gs and gs + gr - 1 <= T,
             f"gradual drift (start {gs!r}, rounds {gr!r}) must fit inside fl.T={T}.")

    # scenarios
    tr, va, te = g("scenarios.train"), g("scenarios.val"), g("scenarios.test")
    for nm, s in (("train", tr), ("val", va), ("test", te)):
        need(isinstance(s, Config), f"scenarios.{nm} must be a mapping.")
        if isinstance(s, Config):
            need(isinstance(s["a"], list) and len(s["a"]) > 0 and all(x > 0 for x in s["a"]),
                 f"scenarios.{nm}.a must be a non-empty list of positive numbers.")
            need(isinstance(s["drift"], list) and len(s["drift"]) > 0 and all(x in DRIFTS for x in s["drift"]),
                 f"scenarios.{nm}.drift must be a non-empty list drawn from {DRIFTS}.")
    if all(isinstance(s, Config) for s in (tr, va, te)):
        ns = tr["n_seeds"]
        need(isinstance(ns, int) and ns >= 1, "scenarios.train.n_seeds must be an int >= 1.")
        vs, ts = list(va["seeds"]), list(te["seeds"])
        need(len(vs) > 0 and len(ts) > 0, "scenarios.val.seeds and scenarios.test.seeds must be non-empty.")
        if isinstance(ns, int):
            train_seeds = set(range(ns))
            need(not (train_seeds & set(vs)) and not (train_seeds & set(ts)) and not (set(vs) & set(ts)),
                 "Train (0..n_seeds-1), validation and test scenario seeds must be disjoint.")
        need(not (set(tr["a"]) & set(va["a"])) and not (set(tr["a"]) & set(te["a"]))
             and not (set(va["a"]) & set(te["a"])),
             "Skew values a must be disjoint across train, validation and test sets.")

    ta = g("eval.target_acc")
    need(isinstance(ta, Config) and all(d in ta for d in DATASETS), "eval.target_acc must define mnist, fmnist, cifar10.")
    need(0 < float(g("eval.ci", 0.95)) < 1, "eval.ci must be in (0, 1).")
    need(int(g("eval.bootstrap_resamples", 1)) >= 1, "eval.bootstrap_resamples must be >= 1.")

    if errs:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errs))


# ----------------------------------------------------------------------------- device
def resolve_device(cfg: Config) -> torch.device:
    """Resolve run.device (auto = cuda if available else cpu) and set the CPU thread count."""
    name = str(cfg.get("run.device", "auto")).lower()
    if name == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(name)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"run.device={name!r} requested but CUDA is not available.")
    torch.set_num_threads(int(cfg.get("run.num_threads", 1)))
    return dev
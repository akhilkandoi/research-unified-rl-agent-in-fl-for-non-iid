"""Run directories, resolved-config saving and config-hash checking."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from fedrl.config import config_hash
from fedrl.utils.io import atomic_json, atomic_write_bytes


class ConfigMismatchError(RuntimeError):
    """Raised when a run folder was created with a different result-affecting config."""


def resolve_runs_dir(cfg: Any = None) -> Path:
    """FEDRL_RUNS_DIR, else cfg.run.runs_dir, else mounted Google Drive, else ./runs."""
    env = os.environ.get("FEDRL_RUNS_DIR")
    if env:
        return Path(env)
    if cfg is not None:
        rd = cfg.get("run.runs_dir", None)
        if rd:
            return Path(rd)
    if os.path.isdir("/content/drive/MyDrive"):
        return Path("/content/drive/MyDrive/fedrl_runs")
    return Path("./runs")


def run_dir(cfg: Any, method: str, variant: str, seed: int) -> Path:
    """{runs_dir}/{dataset}/{method}/{variant}/seed{S}/{config_hash}. Pass the RESOLVED cfg."""
    return (resolve_runs_dir(cfg) / str(cfg.run.dataset) / method / variant
            / f"seed{seed}" / config_hash(cfg))


def save_resolved_config(cfg: Any, path: Path | str) -> None:
    """Atomically write the full resolved config as YAML."""
    atomic_write_bytes(Path(path), yaml.safe_dump(cfg.to_dict(), sort_keys=False).encode("utf-8"))


def check_config_match(rd: Path, cfg: Any) -> None:
    """Raise ConfigMismatchError if rd/meta.json holds a different config hash."""
    meta = Path(rd) / "meta.json"
    if not meta.exists():
        return
    old = json.loads(meta.read_text()).get("config_hash")
    new = config_hash(cfg)
    if old != new:
        raise ConfigMismatchError(
            f"Run folder {rd} was created with config hash {old}, current config hash is {new}. "
            "Use a new variant/runs_dir or restore the original config (run.*, ppo.episodes and "
            "logging.* may change freely).")


def prepare_run_dir(cfg: Any, method: str, variant: str, seed: int,
                    shard: tuple[int, int] | None = None) -> Path:
    """Create the run folder, verify hash, write config.yaml and meta.json."""
    rd = run_dir(cfg, method, variant, seed)
    rd.mkdir(parents=True, exist_ok=True)
    check_config_match(rd, cfg)
    save_resolved_config(cfg, rd / "config.yaml")
    meta_path = rd / "meta.json"
    created = json.loads(meta_path.read_text()).get("created") if meta_path.exists() else time.time()
    atomic_json({"config_hash": config_hash(cfg), "method": method, "variant": variant, "seed": seed,
                 "dataset": str(cfg.run.dataset), "created": created,
                 "shard": list(shard) if shard else None}, meta_path)
    return rd
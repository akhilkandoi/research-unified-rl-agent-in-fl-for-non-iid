"""Deterministic, stream-separated seeding and full RNG state capture/restore."""
from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
import torch


def _digest(seed: int, stream: str) -> bytes:
    return hashlib.sha256(f"{int(seed)}::{stream}".encode("utf-8")).digest()


def stable_seed(seed: int, stream: str) -> int:
    """Stable 63-bit integer derived from (seed, stream); independent of Python's hash()."""
    return int.from_bytes(_digest(seed, stream)[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def make_rng(seed: int, stream: str) -> np.random.Generator:
    """NumPy Generator deterministically derived from (seed, stream)."""
    entropy = int.from_bytes(_digest(seed, stream)[:16], "big")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def make_torch_gen(seed: int, stream: str, device: str | torch.device = "cpu") -> torch.Generator:
    """torch.Generator on `device` deterministically derived from (seed, stream)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(stable_seed(seed, stream))
    return gen


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy (global) and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gen_state(gen: Any) -> dict:
    if isinstance(gen, np.random.Generator):
        return {"type": "numpy", "state": gen.bit_generator.state}
    if isinstance(gen, torch.Generator):
        return {"type": "torch", "state": gen.get_state(), "device": str(gen.device)}
    raise TypeError(f"Unsupported generator type for RNG capture: {type(gen)!r}")


def get_all_rng_states(extra: dict | None = None) -> dict:
    """Capture Python, NumPy global, torch CPU/CUDA states plus named extra generators."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": {name: _gen_state(g) for name, g in (extra or {}).items()},
    }


def _restore_gen(payload: dict, target: Any = None) -> Any:
    if payload["type"] == "numpy":
        if target is None:
            bg = getattr(np.random, payload["state"]["bit_generator"])()
            target = np.random.Generator(bg)
        target.bit_generator.state = payload["state"]
        return target
    if target is None:
        target = torch.Generator(device=payload["device"])
    target.set_state(payload["state"])
    return target


def set_all_rng_states(state: dict, extra: dict | None = None) -> dict:
    """Restore states from get_all_rng_states.

    Returns dict name -> restored generator. If `extra` (name -> generator) is given, those
    objects are restored in place; otherwise new generators are constructed.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    restored = {}
    for name, payload in state.get("extra", {}).items():
        restored[name] = _restore_gen(payload, (extra or {}).get(name))
    return restored
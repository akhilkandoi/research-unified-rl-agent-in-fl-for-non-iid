"""Atomic file writes (temp file in same dir, flush, fsync, os.replace)."""
from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write bytes atomically; on any failure the target is untouched and no temp file remains."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    try:  # best-effort directory sync (not supported on all platforms)
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    """torch.save to memory, then atomic write."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    atomic_write_bytes(path, buf.getvalue())


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def atomic_json(obj: Any, path: str | Path) -> None:
    """Write JSON (sorted keys, indented) atomically."""
    data = json.dumps(obj, indent=2, sort_keys=True, default=_json_default).encode("utf-8")
    atomic_write_bytes(path, data)
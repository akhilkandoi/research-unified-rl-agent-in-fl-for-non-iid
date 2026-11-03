"""Atomic checkpoints: keep last two plus best; fall back if the latest is corrupt."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from fedrl.utils.io import atomic_torch_save
from fedrl.utils.logging_utils import get_logger

log = get_logger(__name__)
_CKPT_RE = re.compile(r"^ckpt_ep(\d+)\.pt$")


class CheckpointManager:
    """Checkpoint state dicts (policy/optimizer/learners, counters, ALL RNG states, best record, config)."""

    def __init__(self, run_dir: Path | str, keep_last: int = 2, map_location: Any = "cpu"):
        # map_location 'cpu' is required so RNG-state ByteTensors stay on CPU;
        # load_state_dict() copies parameters/optimizer state to the target device.
        self.dir = Path(run_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.map_location = map_location

    @property
    def best_path(self) -> Path:
        return self.dir / "best.pt"

    def path_for(self, episode: int) -> Path:
        return self.dir / f"ckpt_ep{episode:06d}.pt"

    def list_checkpoints(self) -> list[tuple[int, Path]]:
        """(episode, path) of regular checkpoints, ascending."""
        out = []
        for p in self.dir.iterdir():
            m = _CKPT_RE.match(p.name)
            if m:
                out.append((int(m.group(1)), p))
        return sorted(out)

    def save(self, state: dict, episode: int, is_best: bool = False) -> None:
        """Atomically save the latest checkpoint (and best.pt if is_best); prune old ones."""
        state = dict(state)
        state["episode"] = int(episode)
        atomic_torch_save(state, self.path_for(episode))
        if is_best:
            atomic_torch_save(state, self.best_path)
        self._prune()

    def save_best(self, state: dict, episode: int) -> None:
        """Atomically save only best.pt."""
        state = dict(state)
        state["episode"] = int(episode)
        atomic_torch_save(state, self.best_path)

    def _prune(self) -> None:
        ckpts = self.list_checkpoints()
        for _, p in ckpts[: max(0, len(ckpts) - self.keep_last)]:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def _try_load(self, path: Path) -> dict | None:
        try:
            try:
                state = torch.load(path, map_location=self.map_location, weights_only=False)
            except TypeError:
                state = torch.load(path, map_location=self.map_location)
            if not isinstance(state, dict) or "episode" not in state:
                raise ValueError("missing 'episode' key")
            return state
        except Exception as exc:  # corrupt / truncated file
            log.warning("Checkpoint %s unreadable (%s); trying previous.", path.name, exc)
            return None

    def load_latest(self) -> dict | None:
        """Newest readable checkpoint, falling back to the previous one; None if none."""
        for _, p in reversed(self.list_checkpoints()):
            state = self._try_load(p)
            if state is not None:
                return state
        return None

    def load_best(self) -> dict | None:
        return self._try_load(self.best_path) if self.best_path.exists() else None
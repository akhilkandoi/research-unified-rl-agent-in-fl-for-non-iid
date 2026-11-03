"""Single-machine lock files with heartbeat + stale detection, and deterministic sharding."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Sequence, TypeVar

T = TypeVar("T")
HEARTBEAT_S = 60.0   # touch interval
STALE_AFTER_S = 900.0  # 15 minutes without heartbeat => stale


class LockHeldError(RuntimeError):
    """Another live process holds the lock."""


def lock_age(path: Path | str) -> float:
    """Seconds since the lock file was last touched."""
    return time.time() - os.stat(path).st_mtime


def is_stale(path: Path | str, stale_after: float = STALE_AFTER_S) -> bool:
    """True if the lock file exists and has not been touched for stale_after seconds."""
    try:
        return lock_age(path) > stale_after
    except FileNotFoundError:
        return False


class RunLock:
    """Exclusive lock file; a daemon thread refreshes its mtime (heartbeat)."""

    def __init__(self, path: Path | str, heartbeat_s: float = HEARTBEAT_S, stale_after: float = STALE_AFTER_S):
        self.path = Path(path)
        self.heartbeat_s = heartbeat_s
        self.stale_after = stale_after
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._token = f"{socket.gethostname()}:{os.getpid()}:{time.time()}"

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if is_stale(self.path, self.stale_after):
                    try:
                        os.remove(self.path)
                    except FileNotFoundError:
                        pass
                    continue
                raise LockHeldError(f"Lock {self.path} is held by a live process "
                                    f"(touched {lock_age(self.path):.0f}s ago).")
            with os.fdopen(fd, "w") as f:
                json.dump({"token": self._token, "host": socket.gethostname(), "pid": os.getpid()}, f)
            break
        else:
            raise LockHeldError(f"Could not acquire lock {self.path}")
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()

    def _beat(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            try:
                os.utime(self.path, None)
            except OSError:
                pass

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            if json.loads(self.path.read_text()).get("token") == self._token:
                os.remove(self.path)
        except (FileNotFoundError, ValueError):
            pass

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def shard_jobs(jobs: Sequence[T], i: int, n: int) -> list[T]:
    """Deterministic, disjoint, covering split: job j belongs to shard j mod n."""
    if n < 1 or not (0 <= i < n):
        raise ValueError(f"Invalid shard {i}/{n}")
    return [j for idx, j in enumerate(jobs) if idx % n == i]


def parse_shard(spec: str | tuple | None) -> tuple[int, int] | None:
    """'2/4' or (2, 4) -> (2, 4); None -> None."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, (tuple, list)):
        i, n = int(spec[0]), int(spec[1])
    else:
        a, b = str(spec).split("/")
        i, n = int(a), int(b)
    shard_jobs([], i, n)  # validates
    return i, n
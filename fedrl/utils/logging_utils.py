"""Logging setup and an append-only CSV logger with truncate-on-resume."""
from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

from fedrl.utils.io import atomic_write_bytes

_CONFIGURED = False


def configure_logging(level: str | int = "INFO") -> None:
    """Configure the 'fedrl' logger once; later calls only change the level."""
    global _CONFIGURED
    root = logging.getLogger("fedrl")
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    root.setLevel(level if isinstance(level, int) else str(level).upper())


def get_logger(name: str) -> logging.Logger:
    """Logger under the 'fedrl' namespace."""
    if not _CONFIGURED:
        configure_logging("INFO")
    return logging.getLogger(name if name.startswith("fedrl") else f"fedrl.{name}")


class CsvLogger:
    """Append-only CSV with a fixed header (written once)."""

    def __init__(self, path: str | Path, fields: list[str]) -> None:
        self.path = Path(path)
        self.fields = list(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "r", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            if header != self.fields:
                raise ValueError(f"CSV header mismatch in {self.path}: file has {header}, expected {self.fields}.")
        else:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.fields)
                f.flush()

    def append(self, row: dict) -> None:
        """Append one row (missing fields are written empty); flushes immediately."""
        extra = set(row) - set(self.fields)
        if extra:
            raise ValueError(f"Unknown CSV fields {sorted(extra)}; allowed: {self.fields}.")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(c, "") for c in self.fields])
            f.flush()

    def truncate_after(self, episode: int) -> None:
        """Drop rows whose 'episode' value is greater than `episode` (atomic rewrite)."""
        if "episode" not in self.fields:
            raise ValueError("CsvLogger.truncate_after requires an 'episode' column.")
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        kept = [r for r in rows if r["episode"] != "" and float(r["episode"]) <= episode]
        buf = io.StringIO(newline="")
        w = csv.writer(buf)
        w.writerow(self.fields)
        for r in kept:
            w.writerow([r.get(c, "") for c in self.fields])
        atomic_write_bytes(self.path, buf.getvalue().encode("utf-8"))
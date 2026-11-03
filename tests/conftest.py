"""Pytest session setup: keep tests away from the real runs directory."""
import os
import sys
from pathlib import Path

os.environ.pop("FEDRL_RUNS_DIR", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
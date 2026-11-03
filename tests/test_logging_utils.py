import csv

import pytest

from fedrl.utils.logging_utils import CsvLogger, get_logger


def _rows(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def test_header_once_append_and_truncate(tmp_path):
    p = tmp_path / "log.csv"
    log = CsvLogger(p, ["episode", "reward"])
    for e in range(1, 6):
        log.append({"episode": e, "reward": e * 0.5})
    assert len(_rows(p)) == 5
    log.truncate_after(3)
    assert [r["episode"] for r in _rows(p)] == ["1", "2", "3"]
    log2 = CsvLogger(p, ["episode", "reward"])  # reopen: no second header
    log2.append({"episode": 4, "reward": 9})
    lines = p.read_text().strip().splitlines()
    assert lines[0] == "episode,reward" and sum(1 for l in lines if l.startswith("episode")) == 1
    assert [r["episode"] for r in _rows(p)] == ["1", "2", "3", "4"]


def test_truncate_to_zero_and_errors(tmp_path):
    p = tmp_path / "log.csv"
    log = CsvLogger(p, ["episode", "x"])
    log.append({"episode": 1, "x": 1})
    log.truncate_after(0)
    assert _rows(p) == []
    with pytest.raises(ValueError, match="Unknown CSV fields"):
        log.append({"episode": 2, "bad": 1})
    with pytest.raises(ValueError, match="header mismatch"):
        CsvLogger(p, ["episode", "y"])


def test_get_logger_namespace():
    assert get_logger("abc").name == "fedrl.abc"
    assert get_logger("fedrl.x").name == "fedrl.x"
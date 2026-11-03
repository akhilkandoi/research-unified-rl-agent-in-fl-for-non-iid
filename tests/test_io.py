import os

import pytest
import torch

import fedrl.utils.io as io_mod
from fedrl.utils.io import atomic_json, atomic_torch_save, atomic_write_bytes


def test_write_and_overwrite(tmp_path):
    p = tmp_path / "sub" / "f.bin"
    atomic_write_bytes(p, b"one")
    atomic_write_bytes(p, b"two")
    assert p.read_bytes() == b"two"
    assert [x.name for x in p.parent.iterdir()] == ["f.bin"]


def test_interrupted_replace_leaves_original_and_no_temp(tmp_path, monkeypatch):
    p = tmp_path / "f.bin"
    atomic_write_bytes(p, b"original")

    def boom(*a, **k):
        raise OSError("simulated crash")

    monkeypatch.setattr(io_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_bytes(p, b"new content")
    monkeypatch.undo()
    assert p.read_bytes() == b"original"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["f.bin"]


def test_failure_during_write_leaves_no_file(tmp_path):
    p = tmp_path / "g.bin"
    with pytest.raises(TypeError):
        atomic_write_bytes(p, "not bytes")  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []


def test_torch_and_json_roundtrip(tmp_path):
    obj = {"w": torch.arange(4.0), "n": 3}
    atomic_torch_save(obj, tmp_path / "c.pt")
    back = torch.load(tmp_path / "c.pt")
    assert torch.equal(back["w"], obj["w"]) and back["n"] == 3
    import json
    import numpy as np
    atomic_json({"a": np.float32(1.5), "b": np.arange(2), "c": torch.tensor([1, 2])}, tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text()) == {"a": 1.5, "b": [0, 1], "c": [1, 2]}
import pytest
import torch

from fedrl.models.cnn import build_model
from fedrl.utils.flat import blend, get_flat, num_params, rel_div, set_flat, weighted_average


def test_roundtrip():
    m = build_model("mnist")
    v = get_flat(m)
    assert v.shape == (num_params(m),)
    new = torch.randn_like(v)
    set_flat(m, new)
    assert torch.equal(get_flat(m), new)
    v[0] = 123.0  # get_flat returns a copy
    assert get_flat(m)[0] != 123.0


def test_set_flat_wrong_size():
    with pytest.raises(ValueError, match="length"):
        set_flat(build_model("mnist"), torch.zeros(5))


def test_weighted_average():
    vecs = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    out = weighted_average(vecs, torch.tensor([1.0, 3.0]))
    assert torch.allclose(out, torch.tensor([2.5, 5.0]))
    with pytest.raises(ValueError):
        weighted_average(vecs, torch.zeros(2))


def test_blend():
    g, l = torch.tensor([1.0, 2.0]), torch.tensor([5.0, 6.0])
    assert torch.equal(blend(1.0, g, l), g)
    assert torch.equal(blend(0.0, g, l), l)
    assert torch.allclose(blend(0.25, g, l), torch.tensor([4.0, 5.0]))
    rows = blend(torch.tensor([1.0, 0.0]), torch.stack([g, g]), torch.stack([l, l]))
    assert torch.equal(rows[0], g) and torch.equal(rows[1], l)


def test_rel_div():
    g = torch.tensor([3.0, 4.0])  # norm 5
    assert rel_div(g, g) == 0.0
    assert rel_div(torch.tensor([3.0, 4.0 + 5.0]), g) == pytest.approx(1.0)
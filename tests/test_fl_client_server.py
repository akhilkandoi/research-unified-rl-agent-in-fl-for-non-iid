import pytest
import torch
import torch.nn.functional as F

from fedrl.fl.client import local_train
from fedrl.fl.fl_round import init_model_vec
from fedrl.fl.server import aggregate
from fedrl.models.cnn import build_model
from fedrl.utils.flat import set_flat, weighted_average
from fedrl.utils.seeding import make_torch_gen


def synthetic(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(10, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    x = protos[y] + 0.3 * torch.randn(n, 1, 28, 28, generator=g)
    return x, y


def loss_of(model, vec, x, y):
    set_flat(model, vec)
    with torch.no_grad():
        return F.cross_entropy(model(x), y).item()


def reference_sgd(model, G, X, y, idx, lr, epochs, batch, gen):
    """Independent plain-SGD implementation with identical shuffling."""
    set_flat(model, G)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    Xc, yc = X[idx], y[idx]
    for _ in range(epochs):
        perm = torch.randperm(len(idx), generator=gen)
        for s in range(0, len(idx), batch):
            b = perm[s:s + batch]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(Xc[b]), yc[b]).backward()
            opt.step()
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def test_aggregate_arithmetic():
    vecs = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    n = torch.tensor([1.0, 3.0])
    G_new, st = aggregate(vecs, n, torch.zeros(3), None, 0.0)
    assert G_new.tolist() == pytest.approx([2.5, 3.5, 4.5])
    assert st is None


def test_server_momentum_zero_equals_plain_average():
    g = torch.Generator().manual_seed(1)
    vecs, n = torch.randn(4, 50, generator=g), torch.tensor([10.0, 20.0, 5.0, 65.0])
    G_new, _ = aggregate(vecs, n, torch.randn(50, generator=g), None, 0.0)
    assert torch.equal(G_new, weighted_average(vecs, n / n.sum()))


def test_server_momentum_formula():
    vecs = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    n = torch.tensor([1.0, 3.0])
    G0 = torch.zeros(3)
    G1, v1 = aggregate(vecs, n, G0, None, 0.5)
    assert G1.tolist() == pytest.approx([2.5, 3.5, 4.5])
    G2, v2 = aggregate(vecs, n, G1, v1, 0.5)          # avg - G1 = 0 -> v = 0.5*v1
    assert v2.tolist() == pytest.approx([1.25, 1.75, 2.25])
    assert G2.tolist() == pytest.approx([3.75, 5.25, 6.75])


def test_local_training_reduces_loss_and_keeps_global():
    x, y = synthetic()
    model = build_model("mnist")
    G = init_model_vec("mnist", 0)
    G_before = G.clone()
    before = loss_of(model, G, x, y)
    w = local_train(model, G, x, y, torch.arange(len(y)), 0.05, 10, 16, 0.0, make_torch_gen(0, "t"))
    assert w.numel() == 46730
    assert loss_of(model, w, x, y) < 0.9 * before
    assert torch.equal(G, G_before)


def test_identical_seeds_identical_results():
    x, y = synthetic()
    model, G, idx = build_model("mnist"), init_model_vec("mnist", 0), torch.arange(100)
    a = local_train(model, G, x, y, idx, 0.01, 2, 32, 0.0, make_torch_gen(5, "s"))
    b = local_train(model, G, x, y, idx, 0.01, 2, 32, 0.0, make_torch_gen(5, "s"))
    c = local_train(model, G, x, y, idx, 0.01, 2, 32, 0.0, make_torch_gen(6, "s"))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_fedprox_mu_zero_equals_plain_sgd_and_mu_pulls_to_global():
    x, y = synthetic()
    model, G, idx = build_model("mnist"), init_model_vec("mnist", 0), torch.arange(120)
    ref = reference_sgd(model, G, x, y, idx, 0.01, 2, 32, make_torch_gen(3, "p"))
    w0 = local_train(model, G, x, y, idx, 0.01, 2, 32, 0.0, make_torch_gen(3, "p"))
    assert torch.equal(w0, ref)
    w_mu = local_train(model, G, x, y, idx, 0.01, 2, 32, 5.0, make_torch_gen(3, "p"))
    assert (w_mu - G).norm() < (w0 - G).norm()


def test_final_partial_batch_is_kept():
    x, y = synthetic(20)
    model, G = build_model("mnist"), init_model_vec("mnist", 0)
    sizes = []
    h = model.register_forward_hook(lambda mod, inp, out: sizes.append(inp[0].shape[0]))
    local_train(model, G, x, y, torch.arange(10), 0.01, 1, 4, 0.0, make_torch_gen(0, "b"))
    h.remove()
    assert sizes == [4, 4, 2]
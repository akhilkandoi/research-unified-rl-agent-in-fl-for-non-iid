"""Tests: Plackett-Luce, masking, attention, encoder."""
import itertools

import pytest
import torch

from fedrl.policy.attention import ClientAttention
from fedrl.policy.encoder import ClientEncoder
from fedrl.policy.masking import (masked_entropy, masked_log_softmax, sample_categorical,
                                  sanitize_option_mask)
from fedrl.policy.plackett_luce import log_prob_of_order, order_to_mask, sample_k_without_replacement


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_pl_probabilities_sum_to_one_N4_k2():
    logits = torch.randn(1, 4, generator=gen())
    orders = torch.tensor(list(itertools.permutations(range(4), 2)))
    assert orders.shape[0] == 12
    mask = torch.ones(12, 4, dtype=torch.bool)
    lp = log_prob_of_order(logits.expand(12, 4), orders, mask)
    assert abs(lp.exp().sum().item() - 1.0) < 1e-5


def test_pl_fewer_eligible_than_k():
    logits = torch.randn(1, 5, generator=gen())
    mask = torch.tensor([[True, False, True, False, False]])
    order, lp, ent = sample_k_without_replacement(logits, mask, 3, gen(1), False)
    assert sorted(order[0, :2].tolist()) == [0, 2] and order[0, 2].item() == -1
    assert torch.isfinite(lp).all() and torch.isfinite(ent).all()
    orders = torch.tensor([[0, 2, -1], [2, 0, -1]])
    lps = log_prob_of_order(logits.expand(2, 5), orders, mask.expand(2, 5))
    assert abs(lps.exp().sum().item() - 1.0) < 1e-5


def test_pl_first_pick_marginal_matches_softmax():
    logits = torch.tensor([[0.5, -0.3, 1.0, 0.0]]).expand(20000, 4)
    mask = torch.ones(20000, 4, dtype=torch.bool)
    order, _, _ = sample_k_without_replacement(logits, mask, 2, gen(3), False)
    freq = torch.bincount(order[:, 0], minlength=4).float() / 20000
    assert torch.allclose(freq, torch.softmax(logits[0], 0), atol=0.02)


def test_pl_masked_clients_never_selected():
    logits = torch.randn(2000, 8, generator=gen())
    mask = torch.ones(2000, 8, dtype=torch.bool)
    mask[:, [1, 4, 6]] = False
    order, _, _ = sample_k_without_replacement(logits, mask, 3, gen(2), False)
    assert (order >= 0).all()
    assert not torch.isin(order, torch.tensor([1, 4, 6])).any()


def test_pl_deterministic_is_topk_and_padding():
    logits = torch.tensor([[0.1, 2.0, -1.0, 1.5, 0.7]])
    mask = torch.tensor([[True, True, True, False, True]])
    order, _, _ = sample_k_without_replacement(logits, mask, 2, None, True)
    assert order[0].tolist() == [1, 4]
    order, _, _ = sample_k_without_replacement(logits, mask, 6, None, True)
    assert order[0].tolist() == [1, 4, 0, 2, -1, -1]
    assert order_to_mask(order, 5)[0].tolist() == [True, True, True, False, True]


def test_masked_log_softmax_no_nan_single_or_no_option():
    logits = torch.randn(3, 4, requires_grad=True)
    mask = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]).bool()
    lp = masked_log_softmax(logits, mask)
    assert not torch.isnan(lp).any()
    assert abs(lp[0, 0].item()) < 1e-6
    ent = masked_entropy(lp, mask)
    assert not torch.isnan(ent).any() and ent[0].item() == 0.0 and ent[1].item() == 0.0
    (lp.masked_fill(~mask, 0).sum() + ent.sum()).backward()
    assert torch.isfinite(logits.grad).all()


def test_epochs_masked_option_never_sampled():
    g = gen()
    logits = torch.randn(4000, 3, 3, generator=g)
    mask = torch.ones(4000, 3, 3, dtype=torch.bool)
    mask[:, :, 2] = False
    mask[:, 1, 0] = False
    idx = sample_categorical(logits, mask, g, False)
    assert not (idx == 2).any() and not (idx[:, 1] == 0).any()
    assert set(idx[:, 0].unique().tolist()) == {0, 1}
    # all-masked rows are sanitised to option 0
    assert sanitize_option_mask(torch.zeros(2, 3, dtype=torch.bool))[:, 0].all()


@pytest.mark.parametrize("N", [3, 20, 50])
def test_attention_equivariant_and_switchable(N):
    att = ClientAttention(16, 4, True).eval()
    x = torch.randn(2, N, 16, generator=gen())
    perm = torch.randperm(N, generator=gen(1))
    assert torch.allclose(att(x)[:, perm], att(x[:, perm]), atol=1e-5)
    off = ClientAttention(16, 4, False)
    assert torch.equal(off(x), x) and len(list(off.parameters())) == 0


def test_encoder_modes_shapes():
    s = torch.randn(2, 5, 24)
    e = ClientEncoder(24, [16, 16], "ema")
    assert e(s)[0].shape == (2, 5, 16) and e(s)[1] is None
    w = ClientEncoder(24, [16, 16], "gru_window", 8)
    out, nh = w(s, window=torch.randn(2, 5, 5, 24))
    assert out.shape == (2, 5, 16) and nh is None
    with pytest.raises(ValueError):
        w(s)
    c = ClientEncoder(24, [16, 16], "gru_const", 8)
    out, nh = c(s, hidden=torch.zeros(2, 5, 8))
    assert out.shape == (2, 5, 16) and nh.shape == (2, 5, 8)
    with pytest.raises(ValueError):
        c(s)
    with pytest.raises(ValueError):
        ClientEncoder(24, [16], "bogus")
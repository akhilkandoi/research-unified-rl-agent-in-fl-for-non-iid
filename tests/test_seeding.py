import random

import numpy as np
import torch

from fedrl.utils.seeding import (get_all_rng_states, make_rng, make_torch_gen, seed_everything,
                                 set_all_rng_states, stable_seed)


def test_streams_reproducible():
    a = make_rng(1, "data").random(5)
    b = make_rng(1, "data").random(5)
    assert np.array_equal(a, b)
    t1 = torch.rand(4, generator=make_torch_gen(1, "x"))
    t2 = torch.rand(4, generator=make_torch_gen(1, "x"))
    assert torch.equal(t1, t2)
    assert stable_seed(3, "s") == stable_seed(3, "s")


def test_streams_independent():
    a = make_rng(1, "data").random(5)
    assert not np.array_equal(a, make_rng(1, "policy").random(5))
    assert not np.array_equal(a, make_rng(2, "data").random(5))
    t1 = torch.rand(4, generator=make_torch_gen(1, "x"))
    assert not torch.equal(t1, torch.rand(4, generator=make_torch_gen(1, "y")))


def test_rng_state_roundtrip():
    seed_everything(0)
    ng = make_rng(5, "n")
    tg = make_torch_gen(5, "t")
    st = get_all_rng_states({"n": ng, "t": tg})
    first = (random.random(), np.random.rand(3), torch.rand(3), ng.random(3), torch.rand(3, generator=tg))
    # perturb, then restore in place
    random.random(); np.random.rand(10); torch.rand(10); ng.random(10); torch.rand(10, generator=tg)
    restored = set_all_rng_states(st, extra={"n": ng, "t": tg})
    second = (random.random(), np.random.rand(3), torch.rand(3), ng.random(3), torch.rand(3, generator=tg))
    assert restored["n"] is ng and restored["t"] is tg
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert np.array_equal(first[3], second[3])
    assert torch.equal(first[4], second[4])


def test_rng_state_restore_constructs_new_generators():
    ng = make_rng(9, "n")
    tg = make_torch_gen(9, "t")
    st = get_all_rng_states({"n": ng, "t": tg})
    exp_n, exp_t = ng.random(4), torch.rand(4, generator=tg)
    restored = set_all_rng_states(st)
    assert np.array_equal(restored["n"].random(4), exp_n)
    assert torch.equal(torch.rand(4, generator=restored["t"]), exp_t)
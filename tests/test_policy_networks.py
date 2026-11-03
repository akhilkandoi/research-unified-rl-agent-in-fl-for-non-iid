"""Tests: JointPolicy and StackedPolicy (not dependent on Part 4; D=24 is a constant)."""
import pytest
import torch

from fedrl.config import load_config
from fedrl.policy.joint_policy import JointPolicy
from fedrl.policy.stacked import StackedPolicy, count_parameters, print_parameter_report

D = 24


def make_cfg(*overrides):
    return load_config(None, list(overrides))


def make_obs(N, seed=0, eligible=None, emask=None, window=False, hidden_dim=0):
    g = torch.Generator().manual_seed(seed)
    elig = torch.ones(N, dtype=torch.bool) if eligible is None else eligible
    if emask is None:
        emask = elig[:, None].expand(N, 3).clone()
    obs = {"state": torch.randn(N, D, generator=g), "eligible": elig, "epochs_mask": emask, "round": 0}
    if window:
        obs["window"] = torch.randn(N, 5, D, generator=g)
    if hidden_dim:
        obs["hidden"] = torch.randn(N, hidden_dim, generator=g)
    return obs


def batch_from(obs, action, k):
    ob = {key: v.unsqueeze(0) for key, v in obs.items() if torch.is_tensor(v)}
    sel = torch.full((1, k), -1, dtype=torch.long)
    sel[0, : len(action["selected"])] = action["selected"]
    ab = {"selected": sel}
    for key in ("lr_idx", "epochs_idx", "batch_idx", "alpha_idx"):
        ab[key] = action[key].unsqueeze(0)
    return ob, ab


def boost_heads(pol):
    """Make logits non-trivial (default head init is near-uniform)."""
    g = torch.Generator().manual_seed(7)
    for p in pol.heads.parameters():
        p.data = torch.randn(p.shape, generator=g) * 0.5


def gen(seed=1):
    return torch.Generator().manual_seed(seed)


# ------------------------------------------------------------ sample == evaluate
@pytest.mark.parametrize("N,k", [(5, 2), (20, 5), (50, 10)])
def test_joint_and_stacked_sample_equals_evaluate_any_N(N, k):
    cfg = make_cfg(f"fl.k={k}")
    for cls in (JointPolicy, StackedPolicy):
        pol = cls(cfg, D).eval()
        obs = make_obs(N)
        out = pol.sample(obs, gen())
        assert len(out.action["selected"]) == k
        assert len(set(out.action["selected"].tolist())) == k
        ob, ab = batch_from(obs, out.action, k)
        with torch.no_grad():
            lp, ent, val = pol.evaluate(ob, ab)
        assert set(lp) == set(pol.agent_names)
        for n in pol.agent_names:
            assert torch.allclose(lp[n][0], out.logp[n], atol=1e-5)
            assert torch.allclose(ent[n][0], out.entropy[n], atol=1e-5)
            assert torch.allclose(val[n][0], out.value[n], atol=1e-5)
            assert torch.isfinite(out.logp[n]) and torch.isfinite(out.entropy[n])


def test_joint_logp_and_entropy_match_hand_computation():
    pol = JointPolicy(make_cfg("fl.k=2"), D).eval()
    boost_heads(pol)
    obs = make_obs(4)
    out = pol.sample(obs, gen())
    act = out.action
    ob, ab = batch_from(obs, act, 2)
    with torch.no_grad():
        logits, _, _ = pol._forward(ob["state"], None, None)
    sel = logits["select"][0]
    lr_l, ep_l, ba_l = (t[0] for t in logits["train"])
    al_l = logits["mix"][0]
    exp_lp, exp_ent = torch.tensor(0.0), torch.tensor(0.0)
    remaining = [0, 1, 2, 3]
    order = act["selected"].tolist()
    for c in order:
        exp_lp = exp_lp + sel[c] - torch.logsumexp(sel[remaining], 0)
        p = torch.softmax(sel[remaining], 0)
        exp_ent = exp_ent - (p * p.log()).sum()
        remaining.remove(c)
    for c in order:  # only CHOSEN clients count
        for lg, idx in ((lr_l, act["lr_idx"]), (ep_l, act["epochs_idx"]),
                        (ba_l, act["batch_idx"]), (al_l, act["alpha_idx"])):
            d = torch.distributions.Categorical(logits=lg[c])
            exp_lp = exp_lp + d.log_prob(idx[c])
            exp_ent = exp_ent + d.entropy()
    assert torch.allclose(out.logp["joint"], exp_lp, atol=1e-5)
    assert torch.allclose(out.entropy["joint"], exp_ent, atol=1e-5)


def test_stacked_agent_logps_match_hand_computation():
    pol = StackedPolicy(make_cfg("fl.k=2"), D).eval()
    for a in pol.agents.values():
        g = torch.Generator().manual_seed(5)
        a.head.lin.weight.data = torch.randn(a.head.lin.weight.shape, generator=g) * 0.5
    obs = make_obs(4)
    out = pol.sample(obs, gen())
    act = out.action
    ob, _ = batch_from(obs, act, 2)
    with torch.no_grad():
        logits, _, _ = pol._forward(ob["state"], None, None)
    order = act["selected"].tolist()
    tr = sum(torch.distributions.Categorical(logits=lg[0, c]).log_prob(idx[c])
             for c in order
             for lg, idx in zip(logits["train"], (act["lr_idx"], act["epochs_idx"], act["batch_idx"])))
    mx = sum(torch.distributions.Categorical(logits=logits["mix"][0, c]).log_prob(act["alpha_idx"][c])
             for c in order)
    assert torch.allclose(out.logp["train"], tr, atol=1e-5)
    assert torch.allclose(out.logp["mix"], mx, atol=1e-5)


# ------------------------------------------------------------ masks
def test_masked_clients_and_epochs_never_chosen_by_policy():
    N = 10
    elig = torch.ones(N, dtype=torch.bool)
    elig[:4] = False
    emask = elig[:, None].expand(N, 3).clone()
    emask[:, 2] = False
    for cls in (JointPolicy, StackedPolicy):
        pol = cls(make_cfg("fl.k=3"), D).eval()
        g = gen()
        for s in range(30):
            out = pol.sample(make_obs(N, s, elig, emask), g)
            sel = out.action["selected"]
            assert (sel >= 4).all()
            assert (out.action["epochs_idx"][sel] != 2).all()


@pytest.mark.parametrize("n_elig", [3, 1, 0])
def test_no_nan_when_k_or_fewer_eligible(n_elig):
    N = 8
    elig = torch.zeros(N, dtype=torch.bool)
    elig[:n_elig] = True
    for cls in (JointPolicy, StackedPolicy):
        pol = cls(make_cfg("fl.k=3"), D)
        obs = make_obs(N, 0, elig)
        out = pol.sample(obs, gen())
        assert len(out.action["selected"]) == n_elig
        for n in pol.agent_names:
            assert torch.isfinite(out.logp[n]) and torch.isfinite(out.entropy[n])
        ob, ab = batch_from(obs, out.action, 3)
        lp, ent, val = pol.evaluate(ob, ab)
        total = sum(lp[n].sum() + ent[n].sum() + val[n].sum() for n in pol.agent_names)
        total.backward()
        for p in pol.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


# ------------------------------------------------------------ inactive heads
def test_inactive_heads_give_defaults_and_zero_logp():
    cfg = make_cfg("policy.active_heads.select=false", "policy.active_heads.train=false",
                   "policy.active_heads.mix=false")
    pol = JointPolicy(cfg, D)
    assert len(pol.heads) == 0
    N = 6
    elig = torch.tensor([True, True, False, True, True, True])
    emask = elig[:, None].expand(N, 3).clone()
    emask[0, 1] = False          # default epochs (idx 1) masked for client 0 -> largest feasible (idx 2)
    emask[3, 1:] = False         # only option 0 feasible for client 3
    out = pol.sample(make_obs(N, 0, elig, emask), gen())
    a = out.action
    assert out.logp["joint"].item() == 0.0 and out.entropy["joint"].item() == 0.0
    assert (a["lr_idx"] == 1).all() and (a["batch_idx"] == 1).all() and (a["alpha_idx"] == 2).all()
    assert a["epochs_idx"][0].item() == 2 and a["epochs_idx"][3].item() == 0 and a["epochs_idx"][1].item() == 1
    assert 2 not in a["selected"].tolist() and len(a["selected"]) == 5
    # inactive selection is random even in deterministic mode (uses the generator)
    s1 = pol.sample(make_obs(N), gen(1), deterministic=True).action["selected"].tolist()
    s2 = pol.sample(make_obs(N), gen(2), deterministic=True).action["selected"].tolist()
    s3 = pol.sample(make_obs(N), gen(1), deterministic=True).action["selected"].tolist()
    assert s1 == s3
    assert len({tuple(pol.sample(make_obs(N), gen(i), True).action["selected"].tolist()) for i in range(20)}) > 1
    del s2


def test_train_head_inactive_only():
    pol = JointPolicy(make_cfg("policy.active_heads.train=false"), D)
    assert "train" not in pol.heads
    out = pol.sample(make_obs(6), gen())
    assert (out.action["lr_idx"] == 1).all() and (out.action["epochs_idx"] == 1).all()
    assert (out.action["batch_idx"] == 1).all()


def test_gradients_reach_only_active_heads():
    pol = JointPolicy(make_cfg("policy.active_heads.train=false"), D)
    obs = make_obs(6)
    out = pol.sample(obs, gen())
    ob, ab = batch_from(obs, out.action, 5)
    lp, ent, val = pol.evaluate(ob, ab)
    (lp["joint"].sum() + ent["joint"].sum() + val["joint"].sum()).backward()
    for h in ("select", "mix"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pol.heads[h].parameters())
    names = {n for n, _ in pol.named_parameters()}
    assert not any(n.startswith("heads.train") for n in names)


def test_stacked_gradients_isolated_per_agent():
    pol = StackedPolicy(make_cfg("fl.k=2"), D)
    obs = make_obs(6)
    out = pol.sample(obs, gen())
    ob, ab = batch_from(obs, out.action, 2)
    lp, _, _ = pol.evaluate(ob, ab)
    lp["train"].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pol.agents["train"].parameters())
    for other in ("select", "mix"):
        assert all(p.grad is None for p in pol.agents[other].parameters())


# ------------------------------------------------------------ structure
@pytest.mark.parametrize("attn", ["true", "false"])
def test_permutation_equivariance_and_value_invariance(attn):
    pol = JointPolicy(make_cfg(f"policy.attention={attn}"), D).eval()
    boost_heads(pol)
    N = 9
    state = torch.randn(1, N, D, generator=gen())
    perm = torch.randperm(N, generator=gen(3))
    with torch.no_grad():
        l1, v1, _ = pol._forward(state, None, None)
        l2, v2, _ = pol._forward(state[:, perm], None, None)
    assert torch.allclose(l1["select"][:, perm], l2["select"], atol=1e-5)
    for a, b in zip(l1["train"], l2["train"]):
        assert torch.allclose(a[:, perm], b, atol=1e-5)
    assert torch.allclose(l1["mix"][:, perm], l2["mix"], atol=1e-5)
    assert torch.allclose(v1, v2, atol=1e-5)


def test_deterministic_mode_is_topk_and_argmax():
    pol = JointPolicy(make_cfg("fl.k=3"), D).eval()
    boost_heads(pol)
    N = 7
    elig = torch.tensor([True, False, True, True, True, True, True])
    emask = elig[:, None].expand(N, 3).clone()
    emask[:, 2] = False
    obs = make_obs(N, 0, elig, emask)
    out = pol.sample(obs, None, deterministic=True)
    with torch.no_grad():
        logits, _, _ = pol._forward(obs["state"][None], None, None)
    sel = logits["select"][0].masked_fill(~elig, float("-inf"))
    assert out.action["selected"].tolist() == sel.topk(3).indices.tolist()
    lr_l, ep_l, ba_l = (t[0] for t in logits["train"])
    assert torch.equal(out.action["lr_idx"], lr_l.argmax(-1))
    assert torch.equal(out.action["batch_idx"], ba_l.argmax(-1))
    assert torch.equal(out.action["alpha_idx"], logits["mix"][0].argmax(-1))
    assert torch.equal(out.action["epochs_idx"][elig], ep_l.masked_fill(~emask, float("-inf")).argmax(-1)[elig])


def test_separate_bodies_have_distinct_parameters():
    pol = JointPolicy(make_cfg("policy.separate_bodies=true"), D)
    assert set(pol.bodies.keys()) == {"select", "train", "mix"}
    ids = [{id(p) for p in pol.bodies[k].parameters()} for k in pol.bodies]
    assert all(ids[i].isdisjoint(ids[j]) for i in range(3) for j in range(i + 1, 3))
    obs = make_obs(6)
    out = pol.sample(obs, gen())
    ob, ab = batch_from(obs, out.action, 5)
    _, _, val = pol.evaluate(ob, ab)
    val["joint"].sum().backward()   # value head uses the selection body only
    assert any(p.grad is not None for p in pol.bodies["select"].parameters())
    for k in ("train", "mix"):
        assert all(p.grad is None for p in pol.bodies[k].parameters())
    shared = JointPolicy(make_cfg(), D)
    assert list(shared.bodies.keys()) == ["shared"]
    assert count_parameters(pol)["total"] > count_parameters(shared)["total"]


def test_stacked_agents_share_no_parameters():
    pol = StackedPolicy(make_cfg(), D)
    ids = [{id(p) for p in pol.agents[k].parameters()} for k in pol.agent_names]
    assert all(ids[i].isdisjoint(ids[j]) for i in range(3) for j in range(i + 1, 3))
    assert sum(len(s) for s in ids) == len(list(pol.parameters()))


# ------------------------------------------------------------ memory modes
@pytest.mark.parametrize("cls", [JointPolicy, StackedPolicy])
def test_gru_window_gradients_reach_gru(cls):
    pol = cls(make_cfg("state.memory_mode=gru_window", "fl.k=2"), D)
    obs = make_obs(6, window=True)
    out = pol.sample(obs, gen())
    ob, ab = batch_from(obs, out.action, 2)
    lp, ent, val = pol.evaluate(ob, ab)
    for n in pol.agent_names:   # same values as at sampling time
        assert torch.allclose(lp[n][0], out.logp[n], atol=1e-5)
    sum(lp[n].sum() + val[n].sum() for n in pol.agent_names).backward()
    grus = [m for m in pol.modules() if isinstance(m, torch.nn.GRU)]
    assert grus and all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in g.parameters())
                        for g in grus)
    bad = {k: v for k, v in ob.items() if k != "window"}
    with pytest.raises(ValueError):
        pol.evaluate(bad, ab)


@pytest.mark.parametrize("cls", [JointPolicy, StackedPolicy])
def test_gru_const_hidden_is_detached_and_stored(cls):
    pol = cls(make_cfg("state.memory_mode=gru_const", "fl.k=2"), D)
    H = pol.hidden_dim
    assert pol.init_hidden(5).shape == (5, H)
    obs = make_obs(5, hidden_dim=H)
    out = pol.sample(obs, gen())
    assert out.hidden.shape == (5, H) and not out.hidden.requires_grad
    obs2 = dict(obs, hidden=out.hidden)          # next step uses the returned state
    assert pol.sample(obs2, gen()).hidden.shape == (5, H)
    ob, ab = batch_from(obs, out.action, 2)
    ob["hidden"] = ob["hidden"].clone().requires_grad_(True)
    lp, ent, val = pol.evaluate(ob, ab)
    for n in pol.agent_names:
        assert torch.allclose(lp[n][0], out.logp[n], atol=1e-5)
    sum(lp[n].sum() + val[n].sum() for n in pol.agent_names).backward()
    assert ob["hidden"].grad is None             # D9: constant at update time
    cells = [m for m in pol.modules() if isinstance(m, torch.nn.GRUCell)]
    assert cells and all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in c.parameters())
                         for c in cells)
    # sampling without hidden falls back to zeros
    no_h = {k: v for k, v in obs.items() if k != "hidden"}
    assert pol.sample(no_h, gen()).hidden.shape == (5, H)


# ------------------------------------------------------------ reporting
def test_parameter_report_runs():
    cfg = make_cfg()
    rep = print_parameter_report(cfg, D)
    assert rep["stacked"]["total"] > rep["joint"]["total"] > 0
    assert set(rep["stacked"]["by_module"]) == {"agent:select", "agent:train", "agent:mix"}
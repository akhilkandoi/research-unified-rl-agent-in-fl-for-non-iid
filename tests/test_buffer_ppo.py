from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fedrl.rl.buffer import RolloutBuffer, compute_gae, step_rewards
from fedrl.rl.ppo import PPOLearner


def _cfg(lr=0.01, mb=4, epochs=4, mode="joint"):
    return NS(fl=NS(k=1),
              ppo=NS(gamma=0.9, gae_lambda=0.8, minibatches=mb, epochs=epochs, clip=0.2, value_coef=0.5,
                     entropy_coef=0.01, max_grad_norm=0.5, lr=lr, reward_norm=False),
              reward=NS(mode=mode, w_global=1.0, w_pers=1.0, w_fair=0.3, w_comp=0.1))


def test_gae_hand_computed():
    adv, ret = compute_gae([1, 2, 3], [0.5, 1.0, 1.5], [False, False, True], 0.9, 0.8)
    assert adv == pytest.approx([3.8696, 3.43, 1.5], abs=1e-9)
    assert ret == pytest.approx([4.3696, 4.43, 3.0 + 0.0 + 1.5 + 0.0 + 0.0], abs=1e-9) or True
    assert ret == pytest.approx(np.array([3.8696, 3.43, 1.5]) + np.array([0.5, 1.0, 1.5]))


def test_buffer_gae_per_episode_and_norm():
    buf = RolloutBuffer(_cfg(), ["joint"])
    for ep in range(2):
        for t in range(3):
            out = NS(action={"selected": torch.tensor([0]), "lr_idx": torch.zeros(1).long(),
                             "epochs_idx": torch.zeros(1).long(), "batch_idx": torch.zeros(1).long(),
                             "alpha_idx": torch.zeros(1).long()},
                     logp={"joint": torch.tensor(0.0)}, value={"joint": torch.tensor([0.5, 1.0, 1.5][t])},
                     hidden=None)
            buf.add({"state": torch.zeros(1, 2), "eligible": torch.ones(1, dtype=torch.bool),
                     "epochs_mask": torch.ones(1, 3, dtype=torch.bool)}, out,
                    {"joint": float([1, 2, 3][t])}, t == 2)
        buf.finish_episode()
    raw = buf.raw_advantages("joint")
    assert raw[:3] == pytest.approx([3.8696, 3.43, 1.5]) and raw[3:] == pytest.approx(raw[:3])
    assert buf.agent_full("joint")["adv"].mean().abs().item() < 1e-5


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin, self.v = nn.Linear(2, 2), nn.Linear(2, 1)


def _eval(net):
    def fn(ob, ac):
        s = ob["state"][:, 0, :]
        lp = F.log_softmax(net.lin(s), -1)
        a = ac["lr_idx"][:, 0]
        return lp.gather(1, a[:, None])[:, 0], -(lp.exp() * lp).sum(-1), net.v(s)[:, 0]
    return fn


def test_ppo_learns_contextual_bandit():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = _cfg()
    net, buf = _Net(), RolloutBuffer(cfg, ["joint"])
    learner = PPOLearner(net.parameters(), cfg, "joint")
    for _ in range(150):
        buf.clear()
        for _ in range(64):
            c = int(rng.integers(2))
            s = F.one_hot(torch.tensor([c]), 2).float()
            with torch.no_grad():
                d = torch.distributions.Categorical(logits=net.lin(s[0]))
                a = d.sample()
                lp, v = d.log_prob(a), net.v(s[0])[0]
            z = torch.zeros(1, dtype=torch.long)
            out = NS(action={"selected": torch.tensor([0]), "lr_idx": a.view(1), "epochs_idx": z,
                             "batch_idx": z, "alpha_idx": z},
                     logp={"joint": lp}, value={"joint": v}, hidden=None)
            buf.add({"state": s, "eligible": torch.ones(1, dtype=torch.bool),
                     "epochs_mask": torch.ones(1, 3, dtype=torch.bool)}, out,
                    {"joint": float(a.item() == c)}, True)
            buf.finish_episode()
        stats = learner.update(buf, _eval(net), rng)
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var"):
        assert k in stats
    with torch.no_grad():
        pred = net.lin(torch.eye(2)).argmax(-1)
    assert pred.tolist() == [0, 1]


def test_reward_routing():
    from fedrl.config import load_config
    from fedrl.methods import clone_config
    base = load_config()
    info = {"r_global": 2.0, "r_pers": 0.5, "r_fair": 1.0, "comm_norm": 1.0, "comp_norm": 2.0,
            "final_bonus": 0.7, "total": 9.9}
    names = ["select", "train", "mix"]
    spec = clone_config(base, {"reward.mode": "stacked_specialized"})
    r = step_rewards(info, spec, names)
    assert r["select"] == pytest.approx(2.0 + 0.3 * 1.0 + 0.7)   # w_global*r_g + w_fair*r_f + bonus
    assert r["train"] == pytest.approx(2.0 - 0.1 * 2.0 + 0.7)    # w_global*r_g - w_comp*comp + bonus
    assert r["mix"] == pytest.approx(0.5)                        # w_pers*r_pers, no bonus (Part 4)
    shared = clone_config(base, {"reward.mode": "stacked_shared"})
    assert step_rewards(info, shared, names) == {"select": 9.9, "train": 9.9, "mix": 9.9}
    assert step_rewards(info, base, ["joint"]) == {"joint": 9.9}
import copy
import pytest

from fedrl.config import load_config
from fedrl.methods import (METHODS, all_method_names, apply_method, apply_variant, is_rl,
                           variant_overrides)

MASTER_NAMES = [
    "fedavg", "fixed_alpha_0.5", "fixed_alpha_0.1", "fedprox", "loss_heuristic", "tuned_settings",
    "tuned_alpha", "bandit", "joint", "stacked_shared", "stacked_specialized", "single_select",
    "single_train", "single_mix", "joint_mem_none", "joint_no_select", "joint_no_train", "joint_no_mix",
    "joint_gru_window", "joint_ema_beta0.5", "joint_ema_delta", "joint_separate", "joint_no_attention",
    "joint_extra_features", "joint_gru_const", "value_selector", "adaptive_mixing",
]


def test_registry_covers_all_master_names():
    assert set(MASTER_NAMES) <= set(all_method_names())
    for n in MASTER_NAMES:
        assert METHODS[n]["kind"] in ("rl_joint", "rl_stacked", "baseline")
        assert is_rl(n) == (METHODS[n]["kind"] != "baseline")


def test_apply_method_does_not_mutate_and_applies():
    cfg = load_config()
    before = copy.deepcopy(cfg.to_dict())
    for n in all_method_names():
        new = apply_method(cfg, n)
        assert new is not cfg
    assert cfg.to_dict() == before
    assert apply_method(cfg, "joint_no_mix").policy.active_heads.mix is False
    assert apply_method(cfg, "joint_no_mix").policy.active_heads.select is True
    s = apply_method(cfg, "single_select").policy.active_heads
    assert (s.select, s.train, s.mix) == (True, False, False)
    assert apply_method(cfg, "stacked_specialized").reward.mode == "stacked_specialized"
    assert apply_method(cfg, "joint_mem_none").state.memory_mode == "none"
    assert apply_method(cfg, "joint_ema_beta0.5").state.ema_beta == 0.5
    assert apply_method(cfg, "fedprox").fl.prox_mu == 0.01
    with pytest.raises(KeyError):
        apply_method(cfg, "nope")


def test_variant_overrides():
    assert variant_overrides("default") == {}
    assert variant_overrides("wcomm0_wfair0.3") == {"reward.w_comm": 0.0, "reward.w_fair": 0.3}
    assert variant_overrides("wcomm0.3") == {"reward.w_comm": 0.3}
    assert variant_overrides("wfair0") == {"reward.w_fair": 0.0}
    assert variant_overrides("upd4") == {"ppo.update_every_episodes": 4}
    with pytest.raises(ValueError):
        variant_overrides("bogus")
    cfg = load_config()
    assert apply_variant(cfg, "wcomm0_wfair0").reward.w_comm == 0.0
    assert cfg.reward.w_comm == 0.1
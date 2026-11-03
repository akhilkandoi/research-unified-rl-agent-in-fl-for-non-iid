import pickle

import pytest

from fedrl.config import Config, ConfigError, config_hash, load_config, parse_value, validate


def test_default_loads_and_validates():
    cfg = load_config()
    validate(cfg)
    assert cfg.fl.N == 20 and cfg.fl.k == 5 and cfg.fl.T == 30
    assert cfg.menus.alpha == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert cfg.get("ppo.lr") == 0.0003
    assert cfg.get("does.not.exist", 42) == 42
    assert cfg.policy.active_heads.train is True
    assert cfg.run.runs_dir is None


@pytest.mark.parametrize("name", ["configs/smoke.yaml", "configs/pilot.yaml"])
def test_override_files_load(name):
    cfg = load_config(name)
    validate(cfg)
    assert cfg.menus.lr == [0.001, 0.01, 0.05]  # untouched default


def test_smoke_values():
    cfg = load_config("configs/smoke.yaml")
    assert (cfg.fl.N, cfg.fl.k, cfg.fl.T) == (6, 2, 5)
    assert cfg.policy.hidden == [16, 16]
    assert cfg.scenarios.val.seeds == [500] and cfg.scenarios.test.seeds == [1000]
    assert cfg.scenarios.val.a == [0.15, 0.7]  # merged, not replaced


def test_pilot_values():
    cfg = load_config("configs/pilot.yaml")
    assert (cfg.fl.N, cfg.fl.k, cfg.fl.T) == (10, 3, 20)
    assert cfg.policy.active_heads.train is False
    assert cfg.policy.active_heads.select is True
    assert (cfg.drift.abrupt_round, cfg.drift.gradual_start, cfg.drift.gradual_rounds) == (10, 10, 5)


def test_override_typing():
    cfg = load_config(overrides=[
        "fl.k=3", "reward.w_comm=0", "reward.w_fair=0.25", "ppo.reward_norm=true",
        "run.runs_dir=null", "menus.lr=[0.001, 0.1]", "menus.default_lr_idx=0",
        "ppo.lr=1e-3", "run.device=cpu", "state.memory_mode=none",
        "policy.active_heads={select: true, train: false, mix: true}",
    ])
    assert cfg.fl.k == 3 and isinstance(cfg.fl.k, int)
    assert cfg.reward.w_comm == 0 and isinstance(cfg.reward.w_comm, int)
    assert cfg.reward.w_fair == 0.25
    assert cfg.ppo.reward_norm is True
    assert cfg.run.runs_dir is None
    assert cfg.menus.lr == [0.001, 0.1]
    assert cfg.ppo.lr == pytest.approx(1e-3) and isinstance(cfg.ppo.lr, float)
    assert cfg.run.device == "cpu"
    assert cfg.state.memory_mode == "none"
    assert cfg.policy.active_heads.train is False and cfg.policy.active_heads.mix is True


def test_parse_value_types():
    assert parse_value("5") == 5
    assert parse_value("0.5") == 0.5
    assert parse_value("false") is False
    assert parse_value("null") is None
    assert parse_value("abc") == "abc"
    assert parse_value("[1, 2]") == [1, 2]
    assert parse_value("{a: 1}") == {"a": 1}


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="Unknown config key 'fl.kk'"):
        load_config(overrides=["fl.kk=3"])
    with pytest.raises(ConfigError, match="a.b.c=value"):
        load_config(overrides=["fl.k"])


def test_hash_stability_and_sensitivity():
    base = config_hash(load_config())
    assert len(base) == 12
    assert config_hash(load_config(overrides=["ppo.episodes=1000"])) == base
    assert config_hash(load_config(overrides=["run.device=cpu"])) == base
    assert config_hash(load_config(overrides=["logging.level=DEBUG"])) == base
    assert config_hash(load_config(overrides=["eval.log_curves=false"])) == base
    assert config_hash(load_config(overrides=["tuning.budget_rounds=100"])) == base
    assert config_hash(load_config(overrides=["fl.k=3"])) != base
    assert config_hash(load_config(overrides=["ppo.lr=0.001"])) != base
    assert config_hash(load_config(overrides=["clients.size_mean=600.0"])) == base  # 600 == 600.0


@pytest.mark.parametrize("override,fragment", [
    ("fl.k=30", "fl.k"),
    ("menus.alpha=[0.5, 1.0]", "menus.alpha"),
    ("menus.alpha=[]", "menus.alpha"),
    ("menus.lr=[]", "menus.lr"),
    ("ppo.checkpoint_every=15", "ppo.checkpoint_every"),
    ("ppo.validate_every=7", "ppo.validate_every"),
    ("state.memory_mode=lstm", "state.memory_mode"),
    ("fl.blend_mode=whenever", "fl.blend_mode"),
    ("reward.mode=triple", "reward.mode"),
    ("run.dataset=svhn", "run.dataset"),
    ("fl.T=10", "drift"),
])
def test_validate_rejects(override, fragment):
    cfg = load_config(overrides=[override], validate_cfg=False)
    with pytest.raises(ConfigError) as ei:
        validate(cfg)
    assert fragment in str(ei.value)


def test_validate_collects_multiple_errors():
    cfg = load_config(overrides=["fl.k=99", "run.dataset=x"], validate_cfg=False)
    with pytest.raises(ConfigError) as ei:
        validate(cfg)
    assert "fl.k" in str(ei.value) and "run.dataset" in str(ei.value)


def test_config_object_behaviour():
    import copy
    cfg = load_config()
    cp = copy.deepcopy(cfg)
    cp.fl.k = 2
    assert cfg.fl.k == 5 and cp.fl.k == 2
    assert pickle.loads(pickle.dumps(cfg)) == cfg
    assert isinstance(cfg.to_dict(), dict) and "fl" in cfg
    assert "Config(" in repr(cfg)
    with pytest.raises(AttributeError, match="no key"):
        _ = cfg.nonexistent
    cfg.set("extra.deep.key", 1)
    assert cfg.get("extra.deep.key") == 1
    assert isinstance(Config({"a": {"b": 1}}).a, Config)
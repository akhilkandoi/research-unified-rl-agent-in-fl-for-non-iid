import inspect
import json

from fedrl.config import load_config
from fedrl.data.scenario_sets import ScenarioSpec
from fedrl.eval import evaluate_policy as EP


def test_idempotent_skip(tmp_path, monkeypatch):
    calls = []

    def fake(controller, spec, eval_seed, cfg, bundle, pools, device, env=None):
        calls.append(1)
        return {"metrics": {"global_test_acc": 0.9}}

    monkeypatch.setattr(EP, "evaluate_controller", fake)
    cfg, spec = load_config(None), ScenarioSpec(0.2, "none", 1000)
    r1, computed1 = EP.evaluate_and_save(None, spec, 1000, cfg, None, None, "cpu", tmp_path, {"method": "m"})
    r2, computed2 = EP.evaluate_and_save(None, spec, 1000, cfg, None, None, "cpu", tmp_path, {"method": "m"})
    assert computed1 and not computed2 and len(calls) == 1
    assert json.loads(EP.result_path(tmp_path, spec, 1000).read_text())["metrics"]["global_test_acc"] == 0.9
    EP.evaluate_and_save(None, spec, 1000, cfg, None, None, "cpu", tmp_path, {"method": "m"}, force=True)
    assert len(calls) == 2


def test_test_reporting_does_not_use_validation_selection():
    src = inspect.getsource(EP.evaluate_controller) + inspect.getsource(EP.evaluate_and_save)
    for forbidden in ("validation_scenarios", "load_validation_result", "select_best_baseline", "evaluate_on_validation"):
        assert forbidden not in src


def test_validation_evaluation_never_touches_test_data():
    src = inspect.getsource(EP.evaluate_on_validation)
    assert "get_official_test" not in src and "test_x" not in src and "personal_test_idx" not in src


def test_run_episode_follows_part7_protocol():
    from fedrl.eval.evaluate_policy import run_episode

    class FakeEnv:
        T = 2

        def reset(self, spec):
            self.t = 0
            return {}

        def step(self, a):
            self.t += 1
            return {}, 0.0, self.t >= 2, {}

    class Ctl:
        seed, bound, observed = None, None, 0

        def __init__(self, name):
            self.name = name

        def reset(self, obs): pass
        def bind_env(self, env, data=None): self.bound = data
        def act(self, obs, rng, deterministic=True): return {}
        def observe(self, *a): self.observed += 1

    cfg, spec = load_config(None), ScenarioSpec(0.2, "none", 1003)
    c1 = Ctl("bandit")
    run_episode(c1, FakeEnv(), spec, 1003, cfg, data=("x", "y"))
    assert c1.seed == 1003 and c1.bound == ("x", "y") and c1.observed == 0  # frozen bandit never learns
    c2 = Ctl("adaptive_mixing")
    run_episode(c2, FakeEnv(), spec, 1003, cfg)
    assert c2.observed == 2
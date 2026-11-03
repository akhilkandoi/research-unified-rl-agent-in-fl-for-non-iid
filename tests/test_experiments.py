import csv

import pytest

from fedrl.config import load_config
from fedrl.eval.metrics import result_path, validation_path
from fedrl.experiments import (Job, count_jobs, dedupe, experiments_in_tier, job_key, job_run_dir,
                               job_status, jobs_for, shard_jobs)


@pytest.fixture(scope="module")
def cfg():
    return load_config(None)


def test_job_counts(cfg):
    c = {e: count_jobs(jobs_for(e, cfg)) for e in ("pilot", "E1", "E2", "E3", "E4", "E5", "E2b", "E6", "E7")}
    assert c["pilot"] == {"train": 13, "tune": 2, "eval": 0, "total": 15}
    assert (c["E1"]["train"], c["E1"]["tune"], c["E1"]["eval"]) == (12, 7, 19 * 45)  # bandit is trained, 3 seeds
    assert (c["E2"]["train"], c["E2"]["eval"]) == (10, 5 * 2 * 15)
    assert (c["E3"]["train"], c["E3"]["eval"]) == (5, 5 * 15)
    assert (c["E4"]["train"], c["E4"]["tune"], c["E4"]["eval"]) == (12, 7, 19 * 15)
    assert c["E5"]["total"] == 0
    assert c["E2b"]["train"] == 16 and c["E2b"]["eval"] == 16 * 15
    assert (c["E6"]["train"], c["E6"]["eval"]) == (0, 3 * 3 * 15)
    assert (c["E7"]["train"], c["E7"]["tune"], c["E7"]["eval"]) == (6, 3, (6 + 3) * 15)


def test_scenario_and_dataset_content(cfg):
    e1 = jobs_for("E1", cfg)
    assert {(j.scenario.a, j.scenario.drift) for j in e1 if j.kind == "eval"} == {
        (a, d) for a in (0.05, 0.2, 0.5) for d in ("none", "abrupt", "gradual")}
    assert all(j.dataset == "fmnist" for j in jobs_for("E4", cfg))
    assert all(j.dataset == "cifar10" for j in jobs_for("E7", cfg))
    assert all(j.eval_overrides == ("fl.N=50", "fl.k=10") for j in jobs_for("E6", cfg))
    assert {j.variant for j in jobs_for("E3", cfg)} == {"default", "wcomm0_wfair0.3", "wcomm0.3_wfair0.3", "wcomm0.1_wfair0", "wcomm0_wfair0"}


def test_tiers(cfg):
    assert set(experiments_in_tier("core")) == {"pilot", "E1", "E2", "E3", "E4", "E5"}
    assert set(experiments_in_tier("extended")) == {"E2b", "E6", "E7"}


def test_dedupe_removes_e3_default_overlap(cfg):
    jobs = jobs_for("E1", cfg) + jobs_for("E3", cfg)
    assert len(dedupe(jobs)) < len(jobs)
    assert len({job_key(j) for j in dedupe(jobs)}) == len(dedupe(jobs))


@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_sharding_covers_every_job_exactly_once(cfg, n):
    jobs = dedupe(jobs_for("E1", cfg) + jobs_for("E2", cfg))
    shards = [shard_jobs(jobs, (i, n)) for i in range(n)]
    keys = [job_key(j) for s in shards for j in s]
    assert sorted(keys) == sorted(job_key(j) for j in jobs) and len(keys) == len(set(keys))


def test_eval_jobs_follow_their_training_run(cfg):
    jobs = jobs_for("E1", cfg)
    shard_of = {}
    for i in range(3):
        for j in shard_jobs(jobs, (i, 3)):
            shard_of.setdefault((j.method, j.train_seed), set()).add(i)
    assert all(len(v) == 1 for v in shard_of.values())


def test_job_status_train_partial_done(cfg, tmp_path):
    c = load_config(None, [f"run.runs_dir={tmp_path}"])
    job = next(j for j in jobs_for("E1", c) if j.kind == "train")
    assert job_status(job, c) == "not_started"
    rd = job_run_dir(job, c)
    rd.mkdir(parents=True, exist_ok=True)

    def write(last):
        with open(rd / "train_log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "return"])
            for e in range(1, last + 1):
                w.writerow([e, 0.0])

    write(10)
    assert job_status(job, c) == "partial"
    write(int(c.ppo.episodes))
    assert job_status(job, c) == "done"


def test_job_status_eval_and_tune(cfg, tmp_path):
    c = load_config(None, [f"run.runs_dir={tmp_path}"])
    ev = next(j for j in jobs_for("E1", c) if j.kind == "eval")
    assert job_status(ev, c) == "not_started"
    p = result_path(job_run_dir(ev, c), ev.scenario, ev.eval_seed, ev.tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")
    assert job_status(ev, c) == "done"
    tune = next(j for j in jobs_for("E1", c) if j.kind == "tune")
    assert job_status(tune, c) == "not_started"
    rd = job_run_dir(tune, c)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "something.json").write_text("{}")
    assert job_status(tune, c) == "partial"
    validation_path(rd).write_text("{}")
    assert job_status(tune, c) == "done"
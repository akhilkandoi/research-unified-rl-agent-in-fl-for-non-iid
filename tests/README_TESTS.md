# Test map (requirement → test)

Run everything with `pytest`. Smoke-config tests use CPU and synthetic images (no download).
`tests/conftest.py` clears `FEDRL_RUNS_DIR` so tests never write to your real runs folder.
`tests/test_spec_coverage.py` is Part 9. The other modules come from Parts 1 to 8.

## Part 9: tests/test_spec_coverage.py

| Requirement | Test name |
|---|---|
| Dirichlet partition, size bounds, pool disjointness | `test_dirichlet_partition_and_size_bounds` |
| Skew parameter a controls concentration | `test_skew_parameter_controls_concentration` |
| Scenario determinism (same spec, same data) | `test_scenario_is_deterministic` |
| Drift flags (none / abrupt / gradual) | `test_drift_flags_follow_spec` |
| Plackett-Luce probabilities sum to 1, eligibility respected | `test_plackett_luce_probabilities_sum_to_one` |
| Blending arithmetic | `test_blend_arithmetic` |
| Sample-count weighted aggregation | `test_weighted_average_arithmetic` |
| Server momentum arithmetic | `test_aggregate_arithmetic_with_server_momentum` |
| Flat vector round trip, 46,730 params | `test_flat_roundtrip_and_param_count` |
| Local training reproducible, FedProx pulls toward G | `test_local_train_changes_weights_and_is_reproducible` |
| Battery drain, eligibility, epochs masks | `test_battery_drain_and_masks` |
| State dim per memory mode (24 / 26 / extra) | `test_state_dim_modes` |
| Running-average (EMA) update, idle/picked counters | `test_running_average_in_state` |
| Memory mode none zeroes memory features | `test_memory_none_zeroes_features` |
| Policy permutation equivariance | `test_policy_permutation_equivariance` |
| sample() log-prob equals evaluate() (joint and stacked) | `test_sample_logp_matches_evaluate` |
| Only chosen clients count in log-prob and entropy | `test_unchosen_clients_do_not_count_in_logp` |
| Masked clients and options never selected | `test_masked_clients_never_selected` |
| Fewer eligible than k selects all eligible | `test_fewer_eligible_than_k` |
| Env reset/step shapes and info keys | `test_env_reset_step_shapes` |
| Free re-probe at drift | `test_env_probe_count_increases_on_drift` |
| Reward = documented weighted sum, final bonus | `test_env_reward_composition` |
| alpha = 0 gives exactly zero personalization gain | `test_alpha_zero_gives_zero_personalization_gain` |
| Env determinism | `test_env_determinism` |
| Baselines respect eligibility and epoch masks (Part 7 protocol) | `test_baselines_respect_masks` |
| PPO learns a toy contextual bandit | `test_ppo_learns_toy_contextual_bandit` |
| 6 episodes = 3 + restart + 3 | `test_resume_equals_uninterrupted` |
| Corrupt latest checkpoint falls back | `test_corrupt_latest_checkpoint_falls_back` |
| Config hash ignores run.*, ppo.episodes, logging.*; schedule check | `test_config_hash_ignores_non_result_keys` |
| No training code imports get_official_test / reads test scenarios (trainer guard exempt) | `test_training_code_has_no_test_set_access_static` |
| Training survives with get_official_test made fatal | `test_training_does_not_touch_test_data_dynamic` |
| Tiny end-to-end train + evaluate | `test_tiny_end_to_end_smoke` |

## Parts 1 to 8: test modules

| Area | Test files |
|---|---|
| Config, seeding, flat vectors, atomic IO, logging, models, evaluate | `test_config.py`, `test_seeding.py`, `test_flat.py`, `test_io.py`, `test_logging_utils.py`, `test_models.py`, `test_evaluate.py` |
| Datasets, pools, partition, drift, scenario, scenario sets | `test_datasets.py`, `test_pools.py`, `test_partition.py`, `test_drift.py`, `test_scenario.py`, `test_scenario_sets.py` |
| Test-set leak guard | `test_no_test_leak.py` |
| FL core (resources, costs, client, server, FLWorld) | `test_fl_resources_costs.py`, `test_fl_client_server.py`, `test_fl_world.py` |
| Env (state, memory, reward, FLEnv) | `test_env_state.py`, `test_env_memory.py`, `test_env_reward.py`, `test_env_fl_env.py` |
| Policy (masking, Plackett-Luce, joint, stacked) | `test_policy_basics.py`, `test_policy_networks.py` |
| Methods registry, buffer, PPO | `test_methods.py`, `test_buffer_ppo.py` |
| Runs (run manager, checkpoints, locks, logs), trainer, resume | `test_runs.py`, `test_trainer.py` |
| Controllers, bandit, tuning | `test_controllers.py`, `test_bandit.py`, `test_tuning.py` |
| Eval metrics, stats, idempotent results, pilot criteria, analysis | `test_eval_metrics.py`, `test_eval_stats.py`, `test_eval_idempotent.py`, `test_pilot_criteria.py`, `test_analysis.py` |
| Experiments, job lists, sharding | `test_experiments.py` |

## Useful commands
```bash
pytest -q                                   # everything
pytest tests/test_spec_coverage.py -q       # Part 9 only
pytest -k "resume or checkpoint" -q         # resume and checkpoint tests
```
Tests written, not executed. Report failures by test name.
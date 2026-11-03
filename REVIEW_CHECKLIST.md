# Review checklist

Run `pytest`, `python scripts/smoke_test.py` and `python scripts/audit.py` first. Tick the last column only after the command or test passes on your machine.

| # | Item | How to verify | Result |
|---|---|---|---|
| 1 | One joint policy: shared body, three heads (select, train, mix), value head | `pytest tests/test_policy_basics.py tests/test_policy_networks.py -q`; `python -c "from fedrl.policy.joint_policy import JointPolicy as P; from fedrl.config import load_config as L; from fedrl.env.state import state_dim as s; c=L(); print([n for n,_ in P(c,s(c)).named_children()])"` | [ ] |
| 2 | Shared encoder, attention across clients, scales with N | `pytest tests/test_spec_coverage.py -k "permutation_equivariance"`; E6 evaluates the policy at N=50 | [ ] |
| 3 | Selection samples k without replacement from N masked scores; training and alpha count only for chosen clients | `pytest tests/test_spec_coverage.py -k "plackett or masked or unchosen or fewer_eligible"` | [ ] |
| 4 | Menus match MASTER 2.6 | `pytest tests/test_config.py tests/test_methods.py -q`; `python scripts/audit.py` (checks e, j) | [ ] |
| 5 | Joint log-probability is the documented sum (select + chosen lr, epochs, batch, alpha) | `pytest tests/test_spec_coverage.py -k "logp_matches or unchosen"`; read the docstring in `fedrl/policy/joint_policy.py` | [ ] |
| 6 | State has current features plus memory; the running average is correct | `pytest tests/test_spec_coverage.py -k "running_average or state_dim or memory_none"`; `pytest tests/test_env_state.py tests/test_env_memory.py -q` | [ ] |
| 7 | Round order follows MASTER 2.5; blending uses the client's own local model (global model for never-trained clients) | `pytest tests/test_fl_world.py tests/test_env_fl_env.py -q`; `pytest tests/test_spec_coverage.py -k "alpha_zero or env_reset"` | [ ] |
| 8 | Reward uses gain over local-only, normalised costs, validation data only | `pytest tests/test_env_reward.py -q`; `pytest tests/test_spec_coverage.py -k "reward_composition or alpha_zero"`; `python scripts/reward_check.py` | [ ] |
| 9 | Episodes, scenario sets and seeds follow MASTER 2.11; no test data in training or checkpoint selection | `pytest tests/test_no_test_leak.py tests/test_scenario_sets.py -q`; `pytest tests/test_spec_coverage.py -k "test_set_access or touch_test_data"`; `python scripts/audit.py` (checks b, c) | [ ] |
| 10 | Baselines include stacked agents with the same state, budget and both reward versions | `pytest tests/test_policy_networks.py tests/test_controllers.py tests/test_env_reward.py tests/test_methods.py -q`; confirm `stacked_shared` and `stacked_specialized` appear in `all_method_names()` and use the same `ppo.episodes` | [ ] |
| 11 | Seeds, mean-alpha logging and selection counts present | `pytest tests/test_runs.py tests/test_env_fl_env.py -q`; inspect the headers of `train_log.csv`, `val_log.csv` and `selection_counts.csv` in a run folder (mean_alpha, selection counts, seed in `meta.json`) | [ ] |

## Before submission
- [ ] Re-check novelty against recent literature (wording: "to our knowledge").
- [ ] Report parameter counts (task model 46,730 for MNIST/FMNIST; policy parameter counts from `print_parameter_report` / `scripts/benchmark.py`).
- [ ] State compute estimates only from `scripts/benchmark.py` output.
- [ ] Report all ties and losses, and the total number of comparisons.
- [ ] Confirm the best non-RL baseline was chosen from validation scenarios only (`validation.json`), never from test results.
- [ ] Include the local-only definition and everything in KNOWN_ISSUES.md.
- [ ] Rerun `pytest`, `smoke_test.py` and `audit.py` on the final commit.
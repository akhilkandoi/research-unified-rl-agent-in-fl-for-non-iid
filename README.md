# FedRL: one joint PPO policy for client selection, training settings and model mixing in federated learning

A single server-side PPO policy decides every round (1) which k of N clients train, (2) each chosen client's learning rate, epochs and batch size, and (3) a blend weight alpha for every client. The question is whether this beats three stacked independent agents and strong non-RL baselines under Non-IID skew and drift. Everything (FL, PPO, bandit) is written from scratch in PyTorch.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch, torchvision, numpy, pyyaml, pandas, matplotlib, tqdm, pytest (scipy optional)
pip install -e .                       # optional
```

## Quick start
```bash
pytest -q
python scripts/smoke_test.py           # offline, synthetic data, a few seconds
python scripts/audit.py
```
No script needs command-line arguments. Edit the `# ===== EDIT HERE =====` block at the top of a script (or `configs/default.yaml`) and run `python scripts/<name>.py`. Optional flags (for example `--shard 0/2`) override the same variables.

## Run order
1. `pytest`
2. `python scripts/smoke_test.py`
3. `python scripts/audit.py`
4. `python scripts/benchmark.py`. Use the measured seconds per round to trim seeds, scenarios and episodes before long runs. Quote compute figures only from this output.
5. `python scripts/reward_check.py` (checks reward scales; a random policy should have normalised cost near 1)
6. `python scripts/pilot.py`. Go/no-go (criteria D12). On failure it prints the ordered remedies. Nothing is extended automatically.
7. Minimum comparison, via the EDIT HERE block of `scripts/run_all.py`: `joint`, `stacked_shared`, `stacked_specialized`, `fedavg`, `fixed_alpha_0.5`, `fixed_alpha_0.1`, `fedprox`, `loss_heuristic`.
8. Add `python scripts/tune_baselines.py` (tuned_settings, tuned_alpha; validation scenarios only, budget at most episodes x T rounds), then the `bandit` method.
9. E2 to E5 via `run_all.py` (ablations, reward sensitivity, Fashion-MNIST, free log-only analyses).
10. Extended experiments (E2b, E6, E7) only if time allows.

Then `python scripts/status.py` (progress) and `python scripts/analyze.py` (tables, plots, bootstrap CIs).

Order rule: `tuned_*` baselines need their `tuning.json` before they are built or evaluated, and the best non-RL baseline is chosen from `validation.json` only (never from test results).

## Colab
```python
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ['FEDRL_RUNS_DIR'] = '/content/drive/MyDrive/fedrl_runs'
!git clone <your repo> && cd fedrl_project && pip install -r requirements.txt
!cd fedrl_project && python scripts/run_all.py
```
If the session dies, rerun the same command. Training resumes from the last checkpoint. Run folders can be copied between laptop and Colab. Results need not be bit-identical across hardware. The dataset cache stays on local disk.

## Splitting work between laptop and Colab
Jobs are assigned deterministically: job i belongs to shard `i mod num_shards`. Set `SHARD = (0, 2)` on the laptop and `SHARD = (1, 2)` on Colab in `run_all.py` (or pass `--shard 0/2` and `--shard 1/2`), then copy the run folders together. No shared-drive locks are used. Per-machine lock files have a 15-minute heartbeat. Evaluation jobs stay on the shard that trained the run.

## Config reference (`configs/default.yaml`)
| Key | Default | Meaning | Placeholder? |
|---|---|---|---|
| run.dataset | mnist | mnist, fmnist or cifar10 | no |
| run.device | auto | cpu, cuda or auto | no |
| run.num_threads | 1 | CPU threads per process | no |
| run.runs_dir | null | run root (see "Reading logs" for the lookup order) | no |
| run.data_dir | ./data | local dataset cache | no |
| fl.N / fl.k / fl.T | 20 / 5 / 30 | clients, selected per round, rounds per episode | no |
| fl.blend_mode | after_aggregation | or before_training; same for all methods in a comparison | no |
| fl.server_momentum | 0.0 | server momentum m | no |
| fl.prox_mu | 0.0 | FedProx mu (the fedprox method sets 0.01) | no |
| fl.log_all_client_gain | true | log personalization gain over all clients (NaN if false) | no |
| clients.size_mean / size_sigma | 600 / 0.5 | lognormal client size | no |
| clients.size_min / size_max | 150 / 2000 | size clip | no |
| clients.val_frac / val_min | 0.25 / 50 | client validation size | no |
| clients.resource_sigma | 0.5 | lognormal sigma of speed and bandwidth | no |
| clients.battery_init_low / high | 0.5 / 1.0 | initial battery range | no |
| clients.min_battery / recharge | 0.05 / 0.02 | battery floor, per-round recharge | no |
| clients.drain_ref, drain_ref_epochs, drain_ref_n | 0.05, 2, 600 | drain = 0.05 per (2 epochs x 600 samples) | no |
| clients.n_transfers_base | 2 | download + upload | no |
| data.pool_seed | 12345 | fixed pool seed | no |
| data.global_val_size | 5000 | balanced global validation set | no |
| data.client_train_frac | 0.8 | client-train share of the remainder | no |
| data.personal_val_own_frac | 0.8 | own-distribution share of the personal validation set | no |
| data.personal_test_size | 200 | personal test set size (evaluation only) | no |
| drift.abrupt_round | 15 | abrupt drift round | no |
| drift.gradual_start / gradual_rounds | 15 / 8 | gradual drift window | no |
| menus.lr / epochs / batch / alpha | [0.001,0.01,0.05] / [1,2,5] / [16,32,64] / [0.1,0.3,0.5,0.7,0.9] | action menus | no |
| menus.default_*_idx | 1, 1, 1, 2 | fixed-baseline defaults (lr, epochs, batch, alpha) | no |
| state.memory_mode | ema | none, ema, ema_delta, gru_window, gru_const | no |
| state.ema_beta | 0.9 | EMA factor | no |
| state.gru_window | 5 | window length for gru_window | no |
| state.extra_features | false | optional extra state features | no |
| state.loss_scale / div_scale / log_clip | 3.0 / 2.0 / 3.0 | fixed state scaling constants | no |
| policy.hidden | [64,64] | encoder widths | no |
| policy.attention / heads | true / 4 | cross-client attention, number of heads | no |
| policy.separate_bodies | false | own encoder per head | no |
| policy.gru_hidden | 64 | GRU hidden size | no |
| policy.active_heads.select/train/mix | true | head switches (ignored by the stacked policy) | no |
| reward.mode | joint | reward routing mode (joint, stacked_shared, stacked_specialized) | no |
| reward.acc_scale | 10.0 | accuracy scale | no |
| reward.w_global / w_pers / w_fair | 1.0 / 1.0 / 0.3 | reward weights | no |
| reward.w_comm / w_comp | 0.1 / 0.1 | cost weights | no |
| reward.final_bonus / cost_clip / fair_percentile | 1.0 / 3.0 / 10 | final bonus, cost clip, fairness percentile | no |
| ppo.lr / gamma / gae_lambda / clip | 3e-4 / 0.97 / 0.95 / 0.2 | PPO core | no |
| ppo.epochs / minibatches | 4 / 4 | updates per rollout | no |
| ppo.entropy_coef / value_coef / max_grad_norm | 0.01 / 0.5 / 0.5 | loss terms, clipping | no |
| ppo.episodes | 400 | training episodes (raise it to extend a run) | yes |
| ppo.update_every_episodes | 2 | episodes per PPO update (pilot sweeps 2/4/8) | yes |
| ppo.checkpoint_every / validate_every | 10 / 20 | multiples of update_every | no |
| ppo.reward_norm | false | running reward normalisation | no |
| scenarios.train | a [0.1,0.3,1.0], drift [none,abrupt], 100 seeds | training scenarios | no |
| scenarios.val | a [0.15,0.7], drift [none,abrupt], seeds 500-502 | validation scenarios | no |
| scenarios.test | a [0.05,0.2,0.5], drift [none,abrupt,gradual], seeds 1000-1004 | test scenarios (never used in training or selection) | no |
| bandit.ridge / ucb_c | 1.0 / 1.0 | LinUCB | no |
| tuning.budget_rounds | null | tuning cap (null = episodes x T) | no |
| eval.bootstrap_resamples / ci | 10000 / 0.95 | paired bootstrap | no |
| eval.target_acc | mnist 0.95, fmnist 0.80, cifar10 0.45 | rounds-to-target thresholds | no |
| eval.log_curves | true | per-round curves | no |
| logging.level | INFO | log level | no |

`configs/smoke.yaml` and `configs/pilot.yaml` hold the overrides of decision D13.

## Methods
- Non-RL: `fedavg`, `fixed_alpha_0.5`, `fixed_alpha_0.1`, `fedprox`, `loss_heuristic`, `tuned_settings`, `tuned_alpha`, `bandit`.
- RL: `joint`, `stacked_shared`, `stacked_specialized`, `single_select`, `single_train`, `single_mix`.
- Joint ablations: `joint_mem_none`, `joint_no_select`, `joint_no_train`, `joint_no_mix`, `joint_gru_window`.
- Extended: `joint_ema_beta0.5`, `joint_ema_delta`, `joint_separate`, `joint_no_attention`, `joint_extra_features`, `joint_gru_const`, `value_selector`, `adaptive_mixing`.

## Variants
A variant is a string of underscore-joined tokens applied after the method: `default` (no change), `upd<n>` (ppo.update_every_episodes), and `wcomm<x>`, `wfair<x>`, `wcomp<x>`, `wpers<x>`, `wglobal<x>` (reward weights). Example: `wcomm0.1_wfair0.3`. Any other string is an error.

## Reading logs
Run folder: `{runs_dir}/{dataset}/{method}/{variant}/seed{S}/{config_hash}/`. The runs_dir lookup order is: env `FEDRL_RUNS_DIR`, then `run.runs_dir`, then `/content/drive/MyDrive/fedrl_runs` if mounted, then `./runs`.
- `config.yaml`, `meta.json`, `train.lock`
- `train_log.csv`, `val_log.csv`, `selection_counts.csv`. Episodes are 1-based. Rows are truncated back to the checkpoint episode on resume. The CSV headers name every column (return, each reward term, mean alpha, validation return).
- `checkpoints/ckpt_ep######.pt` (last two) and `checkpoints/best.pt` (best validation return)
- `results/a{a}_{drift}_s{seed}_e{eval_seed}[_tag].json`: one per scenario and evaluation seed, skipped if present
- `validation.json` (validation score of non-RL baselines) and `tuning.json` (tuned baselines)

Open the CSVs with pandas. `plot_mean_alpha` in `fedrl.runs.train_log` draws the mean-alpha curve.

## Extending a run
Raise `ppo.episodes` (config or EDIT HERE) and rerun the same command. `ppo.episodes`, `run.*` and `logging.*` are excluded from the config hash, so the run continues from its last checkpoint. Any other change gives a different hash, and a hash mismatch in an existing folder is an error.

## Honest reporting rules
- Report only numbers produced by your own scripts. Never quote unmeasured timings, accuracies or compute.
- Report ties and losses, and the full number of comparisons made. "Better" means the paired-bootstrap 95% CI excludes zero.
- Pick the best non-RL baseline once, from validation scenarios, never from test results. Test scenarios are never used for training, tuning or checkpoint selection.
- Rounds within a run are not independent samples. Statistics are across seeds.
- Local-only caveat (must appear in methods text): L_i is the global model fine-tuned by the client for ONE round and overwritten at every participation, not a persistent personal model.
- Novelty wording: "to our knowledge". Re-check against recent literature before submission.
- List the simplifications in KNOWN_ISSUES.md.
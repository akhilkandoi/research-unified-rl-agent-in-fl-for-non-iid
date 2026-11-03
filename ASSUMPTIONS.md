# ASSUMPTIONS

## Decisions D1 to D16 (from the master design)
- D1: Dataset pools use one fixed pool seed (data.pool_seed = 12345), identical for all scenarios and methods.
- D2: Client validation size = round(0.25*n_i), minimum 50. Initial battery U(0.5, 1.0). Resource lognormal median 1.0, sigma 0.5.
- D3: Gradual drift runs rounds 15 to 22; abrupt drift at round 15. Drift is applied at the START of that round, followed by a free re-probe.
- D4: Fairness term uses the clients' client-validation accuracy of the global model (same numbers as the probe).
- D5: Never-trained clients: local-only = personalized = G (gain 0).
- D6: FedAvg and FedProx use alpha 1.0 (bypassing the menu); other fixed baselines use alpha 0.5 unless named otherwise.
- D7: Bandit: one shared linear model, ridge 1.0, ucb_c 1.0 for training, greedy (c=0) at evaluation; every chosen client updated with the shared round reward.
- D8: Entropy for Plackett-Luce selection is the sum of per-step entropies along the sampled path (documented in code).
- D9: gru_const hidden state is detached at PPO update time (no backprop through time).
- D10: Anti-starvation override is NOT implemented (off by default per design).
- D11: Plateau rule (pilot): smallest update-frequency or episode count whose 5-point moving average of validation return stays within one standard error (across seeds) of its final value from then on.
- D12: Pilot pass criteria: (a) last-20% mean validation return exceeds first-20% by more than its standard error across seeds; (b) mean alpha of chosen clients in [0.2, 0.8] and std over episodes > 0.01; (c) >= 80% of clients picked at least once per episode and no client above 3x its uniform share. On failure print ordered remedies: per-client credit assignment, smaller action space, reward rescaling or normalization, longer training, different memory. Never add episodes automatically.
- D13: configs/smoke.yaml and configs/pilot.yaml override sets as in the master design (see the files).
- D14: Evaluation personal-test sampling uses a dedicated RNG stream keyed by (scenario seed, client id, round of evaluation).
- D15: Rounds-to-target = first round where global TEST accuracy >= target (null if never). Evaluation-only code may compute global test accuracy every round for the AUC.
- D16: Results are idempotent: one JSON per (dataset, method, variant, training seed, scenario spec, evaluation seed), skipped if present.

## Part 1 assumptions
- load_config validates by default (validate_cfg=False to skip). Unknown override keys raise ConfigError.
- config_hash excludes: run.*, ppo.episodes, logging.*, tuning.budget_rounds, eval.log_curves. Integer-valued floats are canonicalized to ints before hashing (600 == 600.0). run.dataset is therefore not hashed (it is part of the run-folder path).
- Scientific-notation strings such as 1e-3 (which PyYAML reads as str) are converted to floats in YAML files and overrides.
- set_all_rng_states(state, extra=None) optional `extra` restores generators in place; otherwise new generators are returned.
- per_class_acc returns Tensor[C] on CPU with NaN for absent classes.
- stable_seed(seed, stream) added to utils/seeding.py; DATASET_SPECS added to models/cnn.py.
- init_model_vec does not touch the global torch RNG (uses fork_rng).
- CIFAR-10 CNN has 259,914 parameters.
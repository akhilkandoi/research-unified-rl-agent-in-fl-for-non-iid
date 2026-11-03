# Known issues and deliberate simplifications (mention in the paper)

| ID | Simplification | Effect | Suggested paper wording | Status |
|---|---|---|---|---|
| KI-1 (D8) | Plackett-Luce selection entropy is the sum of per-step entropies along the sampled path, an approximation of the true entropy over ordered subsets | The entropy bonus is a surrogate and may regularise selection differently from the exact entropy | "We approximate the selection entropy by the sum of step-wise entropies along the sampled path." | [ ] reviewed |
| KI-2 (D9) | In `gru_const` the GRU hidden state is stored in the buffer and detached at update time (no backpropagation through time) | Memory is learned only through the current step, so gru_const may under-use history | "The recurrent state is treated as a constant at update time." | [ ] reviewed |
| KI-3 | Local-only model L_i is the global model fine-tuned for one round, overwritten at every participation, not a persistent personal model | The personalization gain measures blending with a one-round fine-tune, not full personalization | "L_i is the global model fine-tuned by the client for one round." | [ ] reviewed |
| KI-4 (D10) | No anti-starvation override | Some clients may rarely be selected, so selection counts and the 3x-uniform-share pilot check matter | "We do not enforce a minimum participation rate." | [ ] reviewed |
| KI-5 | Unit and smoke tests use synthetic images, not MNIST | They check wiring, not accuracy | n/a (internal) | [ ] reviewed |
| KI-6 | The pilot's "stacked pair" is a full 3-agent `stacked_shared`, because `StackedPolicy` ignores `active_heads` | The pilot compares against three agents, not an exact selection+mixing pair | "The pilot baseline uses three stacked agents." | [ ] reviewed |
| KI-7 | After a drift re-probe, `d_probe_loss` and `d_probe_acc` include the drift jump | Deliberate: the policy sees the drift as a change | "Probe deltas across a drift event include the drift." | [ ] reviewed |
| KI-8 | With `blend_mode=before_training`, the local-only reference for the gain is an uncharged shadow run from G with the same seed | The gain is measured against a shadow run | "Under before-training blending the reference is a shadow run from G." | [ ] reviewed |
| KI-9 | Under `stacked_specialized`, select and train get `w_global*final_bonus`, mix gets none | Reward routing differs by agent | "Final-round bonus is routed to the selection and training agents." | [ ] reviewed |
| KI-10 | The env repairs invalid actions (ineligible or duplicate picks, masked epochs become the largest feasible option, flagged in `info["selection_repaired"]`) | Policies and baselines should never need it; check the flag | n/a (internal) | [ ] reviewed |
| KI-11 | `pers_gain_all` is NaN when `fl.log_all_client_gain=false` (null in JSON) | Loggers and analysis must handle it | n/a (internal) | [ ] reviewed |
| KI-12 | With `policy.active_heads.select=false`, selection is seeded-uniform even in deterministic mode | Ablations without selection are random, not arg-max | "Without the selection head, clients are sampled uniformly." | [ ] reviewed |
| KI-13 | Added: ______ | | | [ ] |
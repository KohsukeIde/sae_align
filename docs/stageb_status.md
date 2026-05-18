# Stage B Status

This repo should treat Stage 0 as the channel-adequacy gate and Stage B as a
controlled alignment pilot. Stage C / PSP-like selective prediction is not the
next step until the Stage B confounds below are resolved.

## Current Interpretation

- Stage 0 is pass-to-Stage-B: the primary RGB redundancy controls are
  `noisy_rgb`, `gray_rgb`, and `blur_rgb`; `edge` is a derived diagnostic; and
  `event_response` is diagnostic-only.
- Stage B is partial positive evidence, not a completed ICLR claim.
- `rgb-range` is the strongest current pair, but it must survive action
  controls before it can support the core action-effect alignment claim.
- `local` is action-conditioned and should be interpreted as a diagnostic pair
  until same-action and residualized controls explain or repair its behavior.

## Stage B.1 Controls

The Stage B.1 reports should include:

- observed action-effect kNN overlap by pair and stratum;
- same-action-type restricted kNN;
- same-action-id restricted kNN;
- action-residualized kNN;
- action-only, shuffled-action, shuffled-strata, and shuffled-embedding controls;
- static-vs-action-effect comparison.

Restricted kNN rows should be interpreted using `chance_adjusted_overlap`,
`n_valid_queries`, and `mean_effective_k`, not raw overlap alone. Small
same-action-id groups can make raw overlap look large even when the expected
random overlap is also large.

`action_residualized` is a sensitivity control, not a primary result. Linear
residualization can remove genuine action-effect signal when the physical effect
is correlated with action metadata. Check `action_residualization_diagnostics.csv`
before interpreting residualized overlap.

For static-vs-action-effect comparison, static observations are repeated across
actions for the same state. Static kNN must therefore exclude same-state
neighbors by default when `state_id` is available, otherwise static alignment is
inflated by duplicate state-action rows.

For `local`, even the static observation is action-conditioned because it is a
pre-action patch around the action site. Treat local static rows as an
action-site diagnostic rather than a pure state-only PRH baseline.

## Reproducible Smoke

Run:

```bash
PYTHONPATH=src bash scripts/run_stageb_b1_smoke.sh outputs/stageb_b1_smoke
```

Key reports:

```text
outputs/stageb_b1_smoke/action_effect_knn/reports/
  alignment_by_pair_and_stratum.csv
  same_action_type_restricted_knn.csv
  same_action_id_restricted_knn.csv
  action_residualized_pairwise_stratified_overlap.csv

outputs/stageb_b1_smoke/static_vs_action_effect/reports/
  static_knn.csv
  action_effect_knn.csv
  delta_minus_static_gain.csv
```

These NumPy pilots are CPU jobs and do not require qsub, GPU, or CUDA. Add qsub
scripts only when moving to neural encoders or world-model training.

# Stage B Status

This repo should treat Stage 0 as the channel-adequacy gate and Stage B as a
controlled alignment pilot. Stage C / PSP-like selective prediction is still too
early. Stage B.2 state-level action-effect signature alignment is implemented as
a smoke/pilot path, but the first v1 CPU experiment did not pass the scientific
gate.

## Current Interpretation

- Stage 0 is pass-to-Stage-B: the primary RGB redundancy controls are
  `noisy_rgb`, `gray_rgb`, and `blur_rgb`; `edge` is a derived diagnostic; and
  `event_response` is diagnostic-only.
- Stage B is partial positive evidence, not a completed ICLR claim.
- `rgb-range` is the strongest current pair, but it must survive action
  controls before it can support the core action-effect alignment claim.
- `local` is action-conditioned and should be interpreted as a diagnostic pair
  until same-action and residualized controls explain or repair its behavior.
- Stage B.2 v1 produced a weak `rgb-range` all-state held-out signal, but not a
  regular-vs-blind stratum signal and not a probe-to-heldout transfer signal.

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

## Stage B.2 Next Priority

Stage B.2 should align state-level action-effect signatures, not individual
state-action rows. For each state `s`, estimate `D_m(s)` for channel `m` from a
probe action split, then evaluate cross-channel alignment on a held-out test
action split.

The primary strata should be state-level labels:

- `regular_state`: state `s` has a reliable channel-visible action-effect
  signature for channel `m`;
- `blind_state`: state `s` has a physical action-effect signature but channel
  `m` does not reliably register it.

The probe/test action split is required because the actions used to estimate or
select `D_m(s)` must not be the same actions used to score the alignment. This
keeps Stage B.2 focused on state-level action-effect geometry rather than
same-row action leakage.

Generate Stage B.2 datasets with `--dense-sampling full-states`; random dense
subsets generally do not contain every action for a selected state and therefore
cannot form a complete `D_m(s)` matrix.

For primary held-out action evidence, fit the action-effect encoder only on the
probe action IDs with `scripts/train_transition_encoder.py --train-action-ids`.
`scripts/analyze_state_signature_knn.py` requires this match by default. Use
`--allow-all-action-trained-model` only for explicitly labeled diagnostics,
because otherwise the encoder has already seen the held-out action columns.
The analyzer also checks the encoder's Stage 0 data fingerprint by default; use
`--allow-cross-data-model` only for explicitly labeled diagnostics.

## Stage B.2 v1 CPU Result

The first non-smoke Stage B.2 run is recorded in
`docs/stageb2_v1_cpu_experiment.md`.

Summary:

- local CPU only, no qsub;
- `512` sampled states, `64` actions, `128` complete dense states;
- probe action IDs `0..31`, held-out action IDs `32..63`;
- action-effect and static encoders trained only on probe action IDs;
- `q=0.25` and `q=0.60` state-stratum thresholds evaluated.

Decision:

```text
Stage B.2 v1: weak partial signal, not pass.
Stage C remains blocked.
```

Key `q=0.25` `rgb-range` results:

```text
held-out action-effect all:           overlap 0.1188, adjusted +0.0400
held-out action-effect regular_state: overlap 0.1187, adjusted +0.0400
held-out action-effect blind_state:   overlap 0.1155, adjusted +0.0368
static all:                           overlap 0.0742
delta-minus-static all:               +0.0445
probe-to-heldout all:                 overlap 0.0742, adjusted -0.0045
```

Interpretation:

- `rgb-range` action-effect signature is above static and above the
  action-column shuffled control on all states.
- `regular_state` is not meaningfully above `blind_state`.
- probe-to-heldout transfer is at chance.
- `q=0.60` has zero valid `rgb-range` regular/blind state queries.

The next loop should repair the state-stratum policy and cross-action transfer
metric before scaling or moving to Stage C.

Bootstrap/fraction follow-up:

- `q=0.25` `rgb-range` held-out all overlap has a bootstrap 95% CI of roughly
  `[0.0984, 0.1414]`, while static all is roughly `[0.0602, 0.0891]` and
  action-column shuffled all is roughly `[0.0516, 0.0785]`.
- `q=0.25` `rgb-range` `regular_state` and `blind_state` intervals overlap
  substantially: regular `[0.0750, 0.1750]`, blind `[0.0931, 0.1401]`.
- `rgb-range` `regular_both_fraction` has q90 `0.25` and max `0.3125`, so
  fixed `0.60` regular-state thresholds are not viable for this run.

## Stage B.3 Gate

Stage B.3 is preregistered in `docs/stageb3_preregistration.md`. It is a
framing-decision gate, not a Stage C experiment. The key change from B.2 is that
redundancy probe-to-heldout calibration and non-transductive normalization must
pass before interpreting `rgb-range`. Held-out action-effect is compared to
held-out static and held-out shuffled controls, not probe-action controls.

The first B.3 v1 CPU run is recorded in `docs/stageb3_v1_cpu_experiment.md`.
It is diagnostic-only: redundancy cross-action calibration did not reliably pass
under the bootstrap CI criterion, so Stage C remains blocked and neither Option
2.5 nor Option 3 is promoted yet.

## Stage B.4 Split-Half Reliability Gate

Stage B.4 is recorded in `docs/stageb4_preregistration.md` and
`docs/stageb4_v1_cpu_experiment.md`. It adds a Gate -1 before cross-channel
alignment: same-channel action-effect signatures must be reliable across the
probe/held-out action split.

The B.4 v1 CPU result:

```text
identity same-action: pass
same-channel split-half reliability: fail
same-channel vs action-column-shuffled paired CI: fail
redundancy cross-channel calibration: fail
```

Decision:

```text
Stage B.4 v1: not pass.
Stop at Gate -1b.
Stage C remains blocked.
Option 3 remains only a candidate, not a pivot.
```

The current blocker is not `rgb-range` itself. It is that same-channel
probe/held-out action-effect signatures are not stable enough to make
cross-channel probe-to-heldout alignment interpretable. Next diagnostics should
separate raw-delta signature reliability, random projection, all-action PCA
upper bound, held-out split-half, and tie-jitter sensitivity.

## Stage B.5 Held-Out Same-Action-Set Gate

Stage B.5 is recorded in `docs/stageb5_preregistration.md` and
`docs/stageb5_v1_cpu_experiment.md`. It demotes the B.4 action-subset transfer
gate and asks whether cross-channel action-effect signatures align when both
channels use the same held-out action set.

The B.5 v1 CPU result:

```text
held-out redundancy controls: pass
pca_probe_only / probe_action_type_apply rgb-range: partial positive
raw_delta and random_projection diagnostics: weaker
continuous observability: positive association
```

Decision:

```text
Stage B.5 v1: partial positive, not full pass.
Stage C remains blocked.
Option 3 remains premature.
```

The key new evidence is that B.2's weak `rgb-range` signal is not simply gone:
under `pca_probe_only / probe_action_type_apply`, B.5 mean adjusted overlap is
about `+0.0447` versus the B.2 reference `+0.0400`. The blocker is that this
signal is representation/normalization dependent and `pca_probe_only` remains
tie-heavy.

## Stage B.6 Artifact And Measurement-Primitive Diagnostics

Stage B.6 is preregistered in `docs/stageb6_preregistration.md`. It keeps the
B.5 held-out same-action-set setup but tests whether the weak `rgb-range`
signal is robust to:

```text
k-sweep
tie-jitter
PCA component sweep
pca_probe_only vs pca_all_action diagnostic
raw_delta / random_projection diagnostics
calibrated CKA / RSA / ridge transfer sanity checks
```

The primary replication cell is fixed before running:

```text
pca_probe_only / probe_action_type_apply / d=32 / k=10 / jitter=0
```

This cell can reproduce B.5, but it cannot by itself justify Stage C. B.6
should be interpreted by sign stability and paired CI across the robustness
families. `pca_all_action` is transductive and diagnostic-only.

Current decision before B.6 v1:

```text
Stage C remains blocked.
Binary strata framing remains weak.
Continuous observability is the current working framing, pending B.6.
Option 3 remains premature.
```

B.6 v1 is recorded in `docs/stageb6_v1_cpu_experiment.md`. The full CPU array
completed 16/16 tasks. The primary replication cell
`pca_probe_only / probe_action_type_apply / d=32 / k=10 / jitter=0` reproduced
the B.5 magnitude:

```text
rgb-range adjusted mean: +0.0437
rgb-range adjusted min:  +0.0150
rgb-range positive:      4/4
redundancy positive:     12/12
AE > static CI:          3/4
AE > shuffled CI:        3/4
detect_geom Spearman:    +0.318
```

Across the robustness grid, `pca_probe_only / probe_action_type_apply` stayed
mostly positive for `d=16/32/64/128`, and raw/random diagnostics were positive
but weaker. The decision is:

```text
Stage B.6 v1: partial robust positive.
Stage C: not automatic, but the metric is no longer obviously broken.
Binary strata framing: weak.
Continuous action-effect observability framing: strengthened.
Option 3 remains premature.
```

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
scripts only when moving to neural encoders, world-model training, or multi-seed
CPU sweeps that clearly exceed local interactive runtime.

For the state-level Stage B.2 smoke:

```bash
PYTHONPATH=src bash scripts/run_stageb_b2_smoke.sh outputs/stageb_b2_smoke
```

Key reports:

```text
outputs/stageb_b2_smoke/state_signature_knn/reports/
  primary_state_signature_knn.csv
  state_signature_knn.csv
  heldout_action_signature_knn.csv
  state_strata_fractions.csv
  state_strata_fraction_summary.csv
  state_signature_bootstrap_ci.csv
  state_delta_minus_static_gain.csv
  diagnostic_probe_delta_minus_static_gain.csv
```

`state_delta_minus_static_gain.csv` compares the primary held-out action-effect
signature to the static baseline. The probe-action version is diagnostic-only.
Use `--bootstrap-repeats` to write `state_signature_bootstrap_ci.csv` with
query-bootstrap confidence intervals for the primary and diagnostic controls.

# Stage B.3 Preregistration

Stage B.3 is a framing-decision gate. It is not a Stage C / PSP-like
experiment, and it is not allowed to rescue a preferred story after inspecting
the numbers.

## Purpose

Current status:

- Stage 0 passed channel adequacy.
- Stage B.1 gave row-level partial signal but was action-confounded.
- Stage B.2 v1 gave weak state-level signal: action-effect was above static and
  shuffled on all states, but regular/blind separation and probe-to-heldout
  transfer failed.

Stage B.3 decides which explanation is most consistent with the next evidence:

```text
1. original / continuous-strata framing remains viable;
2. binary strata are too coarse but continuous observability explains alignment;
3. action-effect acts as a broad coupling signal not localized to strata;
4. metric, normalization, encoder, or environment design is still broken.
```

Stage C remains blocked unless the gates below pass.

## Frozen Primary Setup

- Use `--dense-sampling full-states`.
- Exclude `event_response`, `semantic`, and `edge` from primary evidence.
- Use primary channels:

```text
rgb range local noisy_rgb gray_rgb blur_rgb
```

- Primary pair: `rgb-range`.
- Redundancy calibration pairs: `rgb-noisy_rgb`, `rgb-gray_rgb`,
  `rgb-blur_rgb`.
- Diagnostic pairs: `rgb-local`, `range-local`.
- Probe action IDs must be written before encoder training.
- Action-effect and static encoders must both be trained only on the same probe
  action IDs.
- Cross-data and all-action-trained overrides are diagnostic-only.
- Primary normalization modes must be non-transductive:

```text
none
probe_global_apply
probe_action_type_apply
```

Normalization modes that compute held-out statistics from held-out actions are
diagnostic only and cannot pass a scientific gate.

## Required Reports

Stage B.3 must write:

```text
action_split_balance.csv
action_split_assignments.csv
normalization_sweep_knn.csv
normalization_sweep_bootstrap_ci.csv
gate_summary.csv
observability_score_vs_overlap.csv
regular_minus_blind_correlation.csv
score_quantile_knn.csv
state_strata_fraction_summary.csv
stageb3_summary.json
```

## Loopholes Frozen Before Running

- Direct held-out action-effect overlap is not enough. Probe-to-heldout transfer
  must be calibrated first.
- Redundancy controls must transfer across action splits before interpreting
  `rgb-range`.
- The shuffled action-column control must use the same action split as the
  compared action-effect result. Held-out action-effect must be compared to
  held-out shuffled controls.
- Held-out action-effect features must not be normalized using held-out action
  statistics for primary evidence.
- Static baselines compared to held-out action-effect must use held-out static
  action columns, because `local` static observations are action-conditioned.
- Binary `regular_state` / `blind_state` thresholds are not sufficient unless
  valid query counts are nonzero and stable.
- Query-bootstrap intervals do not replace multi-seed or multi-split
  uncertainty.

## Decision Table

| Gate | Evidence | Pass Rule | If Failed |
|---|---|---|---|
| Setup validity | command log, encoder metadata, summary JSON | full-state dense data; no diagnostic-only primary channels; action-effect and static `train_action_ids` exactly match probe actions; held-out actions excluded from fitting; fingerprint checks pass | Run is diagnostic only |
| Action split balance | `action_split_balance.csv` | probe/held-out action type, effect magnitude, location, and detectability summaries are comparable enough to avoid obvious split artifacts | Rebuild split before interpreting |
| Redundancy cross-action calibration | `normalization_sweep_bootstrap_ci.csv`, control `probe_to_heldout_cross`, stratum `all`, redundancy pairs | `rgb-noisy_rgb` and `rgb-gray_rgb` have positive chance-adjusted probe-to-heldout under at least one non-transductive primary normalization; stronger versions require CI lower bound `> 0` | Metric/normalization repair first; do not interpret `rgb-range` transfer |
| Target held-out signal | `normalization_sweep_bootstrap_ci.csv`, pair `rgb-range`, control `action_effect_heldout_signature`, stratum `all` | chance-adjusted CI lower bound `> 0` under a primary normalization | Stage B.3 is negative |
| Static baseline gain | `normalization_sweep_knn.csv`, `gate_summary.csv` | `rgb-range` held-out action-effect exceeds held-out static under the same states, k, split, and normalization | Claim cannot be action-effect specific |
| Shuffled-action control | `normalization_sweep_knn.csv`, control `action_column_shuffled_heldout` | `rgb-range` held-out action-effect exceeds held-out shuffled action-column control | Direct held-out signal may be action-column artifact |
| Probe-to-heldout target transfer | `normalization_sweep_bootstrap_ci.csv`, pair `rgb-range`, control `probe_to_heldout_cross` | chance-adjusted CI lower bound `> 0` under a primary normalization | No Stage B.2/B.3 pass; weak partial at most |
| Regular/blind binary separation | `normalization_sweep_knn.csv` plus bootstrap rows | `regular_state` exceeds `blind_state` with sufficient valid queries | Do not claim binary strata localization |
| Continuous observability | `observability_score_vs_overlap.csv`, `score_quantile_knn.csv` | threshold-free scores such as `detect_geom_rank_mean` or `detect_geom_raw_mean` show positive association with held-out overlap | Option 2.5 unsupported; consider action-coupling or redesign |

## Final Framing Rules

| Outcome | Required Gates | Framing |
|---|---|---|
| Pass toward Stage C design | setup, split, redundancy transfer, target held-out, static gain, shuffled control, target transfer, and binary or continuous observability pass | state-level action-effect alignment is viable |
| Option 2.5 | setup, split, redundancy transfer, target held-out, static gain, shuffled control pass; binary weak; continuous observability positive | continuous action-effect observability |
| Option 3-lite | setup, split, redundancy transfer, target held-out, static gain, shuffled control pass; binary and continuous observability weak | action as broad causal coupler candidate |
| Diagnostic only | setup, split, or redundancy transfer fails | repair metric / normalization / split before scientific interpretation |
| Negative | setup valid but target held-out and controls fail | record negative Stage B.3; Stage C remains blocked |

## Compute Rule

Start local CPU when the run is comparable to or smaller than Stage B.2 v1.
Use qsub CPU nodes for multi-seed grids unless the planned grid is explicitly
expected to finish within roughly 30 minutes locally and the local execution is
recorded as a compute-process deviation. Use qsub for much larger dense-state
sweeps or any run expected to exceed the local interactive budget. GPU is not
needed for the current NumPy/PCA Stage B.3 diagnostics.

qsub is a compute decision only. It must not change the scientific thresholds.

## Review Loop

Before final interpretation, run read-only reviews:

- experiment auditor: checks metric interpretation, leakage risks, controls,
  and literal application of the decision table;
- implementation auditor: checks split/indexing/normalization/static/shuffle
  behavior;
- process auditor: checks preregistration, qsub decision, and whether deviations
  are labeled diagnostic.

Summarize each review before changing the framing.

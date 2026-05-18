# Stage B.3 v1 CPU Experiment

This note records the first Stage B.3 framing-decision gate run. The output
directory is ignored by git, so the setup, aggregate result, and decision are
tracked here.

## Process

- Date: 2026-05-18.
- Output directory: `outputs/stageb_b3_v1_cpu`.
- Compute decision: local CPU, no qsub. This is a small multi-seed grid and was
  run locally because it was expected to stay within the local interactive
  budget. The preregistration's compute rule says qsub should be used for
  multi-seed grids, so this is recorded as a compute-process deviation. It does
  not change the scientific pass/fail thresholds.
- Dataset seeds: `0`, `1`.
- Split seeds: `17`, `29`.
- Scale per dataset: `256` states, `32` actions, grid size `32`, horizon `3`.
- Dense block per dataset: `4096` dense rows, giving `128` complete states
  times `32` actions.
- Probe fraction: `0.5`, balanced by `scripts/make_balanced_action_split.py`.
- Encoders: PCA transition encoders with `32` components, trained only on probe
  action IDs.
- kNN: `k=10`.
- Bootstrap: `200` query-bootstrap repeats.

Run script:

```bash
PYTHONPATH=src bash scripts/run_stageb_b3_v1_cpu.sh outputs/stageb_b3_v1_cpu
```

Aggregate files:

```text
outputs/stageb_b3_v1_cpu/stageb3_v1_gate_summary_primary.csv
outputs/stageb_b3_v1_cpu/stageb3_v1_bootstrap_selected.csv
outputs/stageb_b3_v1_cpu/stageb3_v1_decision_summary.csv
```

## Important Fixes Before This Run

The Stage B.3 run was not allowed to inherit the earlier B.2 loopholes:

- held-out action-effect is compared to held-out shuffled action columns, not
  probe shuffled columns;
- held-out static baselines use held-out static action columns;
- static encoders must match the same probe `train_action_ids` as the
  action-effect encoders;
- state-stratum thresholds are estimated from the probe action rows used to
  define the state labels;
- non-transductive normalization modes are explicitly marked as primary.

## Gate Results

The key preregistered gate is redundancy cross-action calibration. It did not
reliably pass under the stricter bootstrap-CI interpretation. Some adjusted
means are positive, but the CI lower bounds do not consistently clear zero.

For non-transductive primary normalization modes, redundancy probe-to-heldout
CI lower-bound pass counts were:

```text
rgb-noisy_rgb:
  none:                  0 / 4
  probe_global_apply:    0 / 4
  probe_action_type_apply: 0 / 4

rgb-gray_rgb:
  none:                  1 / 4
  probe_global_apply:    1 / 4
  probe_action_type_apply: 0 / 4
```

The mean adjusted probe-to-heldout values for those redundancy pairs were small
and unstable:

```text
rgb-noisy_rgb:
  none:               +0.0045
  probe_global_apply: +0.0048
  probe_action_type_apply: +0.0003

rgb-gray_rgb:
  none:               +0.0086
  probe_global_apply: +0.0050
  probe_action_type_apply: -0.0039
```

Because redundancy controls do not reliably transfer from probe to held-out
actions under the CI criterion, the Stage B.3 decision table says not to
interpret `rgb-range` cross-action failure as scientific evidence about the
hypothesis.

`rgb-range` still showed a weak held-out action-effect signal:

```text
rgb-range held-out adjusted mean:
  none:               +0.0263
  probe_global_apply: +0.0285
  probe_action_type_apply: +0.0408

rgb-range static adjusted mean:
  none:               +0.0025
  probe_global_apply: +0.0080
  probe_action_type_apply: +0.0080

rgb-range held-out shuffled adjusted mean:
  none:               +0.0029
  probe_global_apply: +0.0066
  probe_action_type_apply: +0.0058
```

That supports the earlier weak observation that action-effect signatures contain
more cross-channel structure than static or shuffled controls. It is not enough
to pass Stage B.3 because probe-to-heldout calibration failed.

`rgb-range` probe-to-heldout remained weak:

```text
mean adjusted:
  none:               -0.0013
  probe_global_apply: +0.0027
  probe_action_type_apply: -0.0019

CI lower-bound pass count:
  none:               0 / 4
  probe_global_apply: 0 / 4
  probe_action_type_apply: 0 / 4
```

## Continuous Observability

The threshold-free detectability geometric mean in rank space showed a positive
association with `rgb-range` held-out overlap across all four data/split
conditions:

```text
detect_geom_rank_mean Spearman, action_effect_heldout_signature:
  none:               mean +0.349
  probe_global_apply: mean +0.304
  probe_action_type_apply: mean +0.310
```

The binary-derived `regular_minus_blind` score was much weaker:

```text
regular_minus_blind Spearman:
  none:               mean +0.030
  probe_global_apply: mean +0.100
  probe_action_type_apply: mean +0.063
```

This is suggestive for a future continuous-observability analysis, but it is
not a framing decision because redundancy cross-action calibration did not
reliably pass.

## Decision

Using `docs/stageb3_preregistration.md`, the decision is:

```text
Stage B.3 v1: diagnostic only.
Primary blocker: redundancy cross-action calibration did not reliably pass under
the bootstrap CI criterion.
Stage C / PSP-like baselines: still blocked.
Option 2.5 / Option 3 pivot: not yet allowed.
```

Current evidence:

- Action-effect held-out signatures are weakly above static and held-out
  shuffled controls for `rgb-range` in most primary summaries, but this is not
  promoted because the redundancy transfer gate fails.
- Probe-to-heldout transfer is not calibrated even for redundancy controls.
- Binary regular/blind localization is still weak.
- Continuous detectability rank product is promising but cannot be promoted
  until metric calibration is fixed.

## Next Loop

Do not scale Stage C. Fix redundancy cross-action calibration first.

Candidate fixes:

1. Add raw-delta and random-projection diagnostics to test whether PCA is the
   cause of redundancy transfer failure.
2. Add an all-action PCA diagnostic upper bound, explicitly labeled non-primary.
3. Add paired bootstrap deltas for held-out action-effect versus static and
   held-out shuffled controls.
4. Revisit probe-to-heldout definition: current kNN cross-transfer may be too
   strict or mismatched for deterministic RGB-derived controls.
5. Defer HOSVD/tensor factorization until raw/random/all-action diagnostics
   identify whether the failure is representation, normalization, or metric
   design.

## Review Loop

The Stage B.3 loop used read-only subagents before and after the run. Their
main findings were:

- design audit: all-state held-out signal must not pass unless redundancy
  probe-to-heldout transfer works first;
- implementation audit: B.2 had high-risk loopholes in probe-vs-heldout shuffled
  controls, transductive normalization, full-action thresholding, static
  baseline matching, static encoder validation, and cross-split query counts;
- alternative representation audit: run all-action PCA, random projection, and
  raw-delta diagnostics before considering HOSVD;
- experiment-size audit: the local CPU 2 data-seed by 2 split-seed grid is
  informative and small enough to run locally, with qsub reserved for larger
  multi-seed sweeps;
- final audit: the decision is supported only as diagnostic-only under the
  bootstrap-CI interpretation, and wording should avoid claiming a clean Stage
  B.3 failure of the scientific hypothesis.

Implementation issues found by the final audit were fixed before the final
aggregate summary was regenerated:

- held-out shuffled controls now use the same probe-fitted normalization
  reference as held-out action-effect features;
- `probe_action_type_apply` now fails if held-out action types are absent from
  probe actions.

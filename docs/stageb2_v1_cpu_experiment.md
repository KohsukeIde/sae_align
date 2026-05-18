# Stage B.2 v1 CPU Experiment

This note records the first non-smoke Stage B.2 state-level action-effect
signature experiment. The output directory is ignored by git, so the key setup,
results, and decision are tracked here.

## Process

- Date: 2026-05-18.
- Output directory: `outputs/stageb_b2_v1_cpu`.
- Compute decision: local CPU, no qsub. The run completed in roughly 10 minutes,
  which is below the current threshold for reserving CPU nodes.
- Scale: `512` sampled states, `64` actions, grid size `32`, horizon `3`.
- Dense Stage B.2 block: `8192` dense rows, giving `128` complete states times
  `64` actions.
- Probe actions: action IDs `0..31`.
- Held-out actions: action IDs `32..63`.
- Encoders: PCA transition encoders with `32` components, trained only on probe
  action IDs via `--train-action-ids`.
- State-level thresholds tested: `0.60` and `0.25`.

The relevant reports are:

```text
outputs/stageb_b2_v1_cpu/state_signature_knn_q060/reports/
outputs/stageb_b2_v1_cpu/state_signature_knn_q025/reports/
```

## Subagent Review

Two read-only subagents independently reviewed the run and repo state without
using web search.

- Experiment auditor: Stage B.2 does not pass the `rgb-range` state-level
  action-effect claim. Direct held-out alignment is slightly above chance, but
  probe-to-heldout transfer is at chance and the strict `q=0.60` strata are not
  usable.
- Process auditor: implementation guardrails are mostly present, but the run
  and confidence loop needed a tracked note. This file is the record of that
  loop.

## Key Results

At `q=0.25`, `rgb-range` on the primary held-out action signature:

```text
all:           overlap 0.1188, chance-adjusted +0.0400, n=128
regular_state: overlap 0.1187, chance-adjusted +0.0400, n=32
blind_state:   overlap 0.1155, chance-adjusted +0.0368, n=116
```

Static versus held-out action-effect for `rgb-range`:

```text
static overlap:        0.0742
action-effect overlap: 0.1188
gain:                 +0.0445
```

The `rgb-range` action-column shuffled control was lower than the held-out
signature:

```text
action-column shuffled all: overlap 0.0648, chance-adjusted -0.0139
```

However, the probe-to-heldout cross-action test failed:

```text
rgb-range all:           overlap 0.0742, chance-adjusted -0.0045
rgb-range regular_state: overlap 0.0688, chance-adjusted -0.0100
rgb-range blind_state:   overlap 0.0750, chance-adjusted -0.0037
```

At `q=0.60`, `rgb-range` had zero valid `regular_state` and `blind_state`
queries, so that threshold is not confirmatory for state strata in this run.

After adding query-bootstrap confidence intervals (`500` repeats), the
`q=0.25` `rgb-range` rows were:

```text
held-out all:           mean 0.1188, 95% CI [0.0984, 0.1414]
held-out regular_state: mean 0.1188, 95% CI [0.0750, 0.1750]
held-out blind_state:   mean 0.1155, 95% CI [0.0931, 0.1401]
probe-to-heldout all:   mean 0.0742, 95% CI [0.0646, 0.0846]
static all:             mean 0.0742, 95% CI [0.0602, 0.0891]
shuffled all:           mean 0.0648, 95% CI [0.0516, 0.0785]
```

The `rgb-range` state-fraction summary explains why fixed `0.60` state strata
collapse:

```text
regular_both_fraction: mean 0.2036, q75 0.2266, q90 0.2500, max 0.3125
blind_either_fraction: mean 0.3149, q75 0.3438, q90 0.4062, max 0.4688
physical_nonnull_fraction: mean 0.5186, q75 0.5625, q90 0.5938, max 0.6875
```

This means a fixed `0.60` `regular_state` threshold is not viable for the
current ToyPowderWorld action bank and channel definitions. A percentile-based
or pair-specific state-stratum policy must be tested before treating state
strata as confirmatory evidence.

Redundancy controls behaved correctly for direct held-out signatures:

```text
rgb-noisy_rgb: overlap 0.7875, chance-adjusted +0.7088
rgb-gray_rgb:  overlap 0.8016, chance-adjusted +0.7228
rgb-blur_rgb:  overlap 0.3328, chance-adjusted +0.2541
```

But the same redundancy controls also fell to chance in the probe-to-heldout
cross-action metric, which means that cross-action transfer or the signature
construction needs more work before it can support the main claim.

## Decision

Stage B.2 v1 is a real experiment, not just implementation. The decision is:

```text
Stage 0: pass
Stage B.1: action-confounded partial positive
Stage B.2 v1: weak partial signal, not pass
Stage C / PSP-like baselines: blocked
```

The `rgb-range` held-out action-effect signature is above static and above the
action-column shuffled control on all states. That is useful signal. It is not
enough for the ICLR claim because:

- `regular_state` is not clearly better than `blind_state`;
- probe-to-heldout transfer is at chance;
- the stricter `q=0.60` state strata are mostly empty;
- the run is one seed, and the current confidence intervals are query-bootstrap
  intervals rather than multi-seed uncertainty.
- bootstrap intervals show large regular/blind overlap at `q=0.25`.

## Confidence Loop

Current confidence is high in the decision not to proceed to Stage C. Current
confidence is not high enough in the Stage B.2 scientific claim.

Remaining loopholes:

- Direct held-out overlap may still be exploiting shared held-out action
  columns rather than transferable state geometry.
- Fixed state-stratum thresholds are unstable; `q=0.60` collapses the main
  strata and `q=0.25` gives weak regular-vs-blind separation.
- Static gain is not selective for `rgb-range`; some diagnostic pairs gain more.
- The run is single seed and still lacks multi-seed uncertainty estimates.
- Probe-to-heldout fails even for redundancy controls, suggesting the
  cross-action evaluation or signature normalization needs repair before
  interpreting `rgb-range`.

Fixes before Stage C:

1. Make probe-to-heldout the primary pass metric and require redundancy controls
   to pass it first.
2. Keep query-bootstrap confidence intervals and add at least a small multi-seed
   run.
3. Report threshold sweeps with valid-query counts and choose a stable
   state-stratum policy before treating strata as evidence.
4. Inspect `state_strata_fractions.csv` and consider percentile-defined
   `regular_state` / `blind_state` labels.
5. Scale dense states only after the redundancy controls transfer across action
   splits. Use qsub CPU nodes for multi-seed or much larger sweeps.

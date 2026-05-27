# Stage B6R Real Powderworld V1 CPU Experiment

Date: 2026-05-27 JST

## Purpose

Stage B6R is the preregistered follow-up to the real-Powderworld B6
static-control postmortem.

It tests the revised Framing D-prime question:

```text
Does action-effect alignment contain intervention-specific cross-modal
structure that survives static controls?
```

It does not test or authorize:

```text
AERA implementation
PSP / Dreamer comparison
Path-Building / Sand-Pushing audit
scalar observability weighting
```

## Preregistration

Rules:

```text
docs/stageb6r_preregistration.md
```

Primary fixed cell:

```text
output root: outputs/stageb6r_realpw_v1_cpu
data seeds: 10, 11, 12
split seeds: 101, 103, 107
PCA dim: 32
normalization: probe_action_type_apply
k: 10
jitter: 0
target pair: rgb:range
runs: 9
```

## Commands

Generation:

```bash
OUT=outputs/stageb6r_realpw_v1_cpu \
DATA_SEEDS='10 11 12' \
N_STATES=128 \
K_ACTIONS=32 \
GRID_SIZE=64 \
HORIZON=4 \
CHANNELS='rgb range noisy_rgb gray_rgb blur_rgb' \
WALLTIME=04:00:00 \
bash scripts/submit_stageb_b6_realpw_generate_array.sh
```

Analysis:

```bash
OUT=outputs/stageb6r_realpw_v1_cpu \
DATA_SEEDS='10 11 12' \
SPLIT_SEEDS='101 103 107' \
PCA_DIMS='32' \
K_VALUES='10' \
JITTER_EPSILONS='0' \
JITTER_SEEDS='0' \
NORMALIZATION_MODES='probe_action_type_apply' \
REPRESENTATIONS='pca_probe_only' \
TARGET_PAIRS='rgb:range' \
WALLTIME=04:00:00 \
bash scripts/submit_stageb_b6_realpw_analysis_array.sh
```

Static controls:

```bash
PYTHONPATH=src bash scripts/run_stageb6r_static_controls.sh \
  outputs/stageb6r_realpw_v1_cpu
```

## Completion

Completed:

```text
datasets: 3/3
B6 analysis reports: 9/9
static-control reports: complete
```

## Original Unconditioned B6 Rows

Primary `rgb:range` adjusted overlap:

```text
action-effect: +0.0237, positive 9/9
static:        +0.0994, positive 9/9
shuffled:      -0.0009, positive 3/9
```

Paired comparisons:

```text
AE > static CI:   0/9
AE > shuffled CI: 6/9
```

Redundancy controls:

```text
positive: 27/27
```

Observability:

```text
detect_geom Spearman mean: +0.3118
```

Interpretation:

```text
real-Powderworld action-effect alignment remains above shuffled, but raw static
alignment is still much stronger.
```

## Static-Residualized kNN

Primary residualizer:

```text
static_residualized_probefit:
  adjusted mean: +0.0293
  adjusted min:  +0.0072
  positive:      9/9
```

Matched shuffled null:

```text
static_residualized_shuffled_probefit:
  adjusted mean: +0.0030
  adjusted min:  -0.0045
  positive:      6/9
```

Difference:

```text
residualized minus shuffled mean: +0.0263
```

Diagnostic residualizers:

```text
static_residualized_crossfit: mean +0.0063
static_residualized_global:   mean +0.0038
```

Residual energy was not near zero:

```text
probe-fit residual norm fraction:
  rgb:   mean 1.2452, min 1.0087
  range: mean 1.5557, min 1.1905
```

## Static-Conditioned kNN

Action-effect inside static-similarity bins:

```text
highest:  +0.0417, positive 8/9
high_mid: +0.0512, positive 9/9
low_mid:  +0.0553, positive 9/9
lowest:   +0.0510, positive 9/9
```

Shuffled inside bins:

```text
highest:  -0.0004
high_mid: -0.0030
low_mid:  +0.0049
lowest:   -0.0100
```

## Literature Metrics

Lightweight calibration used 10 permutations.

Primary residualized CKNNA:

```text
static_residualized_probefit:
  calibrated mean: +0.0365
  calibrated min:  +0.0170
  positive:        9/9
```

Residualized shuffled CKNNA:

```text
static_residualized_shuffled_probefit:
  calibrated mean: +0.0016
  calibrated min:  -0.0161
  positive:        6/9
```

Secondary cycle-kNN:

```text
static_residualized_probefit:
  calibrated mean: +0.0169
  calibrated min:  +0.0032
  positive:        9/9
```

## Preregistered Gate Decision

Strict B6R decision:

```text
No-go / near-miss.
```

Reasons:

```text
Pass: residualized action-effect is positive in 9/9 runs.
Fail: residualized adjusted mean is +0.0293, below the +0.03 threshold.
Fail: residualized minus shuffled mean is +0.0263, below the +0.03 threshold.
Pass: residualized CKNNA mean is +0.0365 and positive in 9/9 runs.
Pass: static-conditioned means are positive in all four bins.
Pass: residual energy is not near zero.
```

Therefore:

```text
B6R supports weak non-static action-effect structure, but it does not pass the
precommitted threshold for opening AERA implementation.
```

## Interpretation

Supported:

```text
real-Powderworld action-effect alignment is not pure shuffled noise;
residualized and static-conditioned diagnostics remain directionally positive.
```

Not supported under the preregistered gate:

```text
Framing D-prime is strong enough to proceed to AERA implementation.
```

Blocked:

```text
AERA implementation
Action-IV Step 3
Path-Building / Sand-Pushing audit
PSP / Dreamer comparison
custom simulator route
```

## Output Files

```text
outputs/stageb6r_realpw_v1_cpu/stageb6_primary_cell_summary.csv
outputs/stageb6r_realpw_v1_cpu/stageb6_decision_summary.csv
outputs/stageb6r_realpw_v1_cpu/stageb6_knn_sensitivity.csv
outputs/stageb6r_realpw_v1_cpu/static_controls_b6r_v1/stageb6_static_control_summary.csv
outputs/stageb6r_realpw_v1_cpu/static_controls_b6r_v1/stageb6_static_residualized_knn.csv
outputs/stageb6r_realpw_v1_cpu/static_controls_b6r_v1/stageb6_static_conditioned_knn.csv
outputs/stageb6r_realpw_v1_cpu/static_controls_b6r_v1/stageb6_static_control_literature_metrics.csv
outputs/stageb6r_realpw_v1_cpu/static_controls_b6r_v1/stageb6_static_residual_diagnostics.csv
```

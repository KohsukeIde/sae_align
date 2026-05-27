# Stage B.6 Static-Control Postmortem

Date: 2026-05-27

## Purpose

This is a post-v1 explanatory diagnostic after real Powderworld B6 showed:

```text
action-effect > shuffled
static > action-effect
```

It asks whether the action-effect signal is fully explained by static state
similarity, or whether an additional action-conditioned structure remains after
controlling for static features.

This diagnostic does **not** retroactively make real Powderworld B6 v1 a
Framing-D pass and does **not** authorize AERA implementation.

## Inputs

Root:

```text
outputs/stageb_b6_realpw_v1_cpu
```

Primary-filtered analysis:

```text
pca_dim: 32
normalization: probe_action_type_apply
k: 10
target pair: rgb:range
runs: 3 data seeds x 3 split seeds = 9
```

Command:

```bash
PYTHONPATH=src python scripts/analyze_stageb6_static_controls.py \
  --root outputs/stageb_b6_realpw_v1_cpu \
  --out outputs/stageb_b6_realpw_v1_cpu/static_controls_primary_lit \
  --expected-report-dirs 9 \
  --pca-dims 32 \
  --channels rgb range \
  --normalization-modes probe_action_type_apply \
  --k-values 10 \
  --target-pairs rgb:range \
  --max-states 128 \
  --permutation-repeats 10
```

## Controls

### Static-Residualized

Primary residualizer:

```text
static_residualized_probefit
```

It fits a same-channel static-to-action-effect ridge residualizer on probe
action columns and applies it to held-out action columns.

Diagnostic residualizers:

```text
static_residualized_global
static_residualized_crossfit
```

These are not primary because they use the evaluation state/action feature
geometry more directly.

### Static-Conditioned kNN

For each query state, candidate neighbors are divided by average same-channel
static similarity into four bins.  Action-effect neighbor overlap is measured
inside each bin with a query-local chance baseline.

## Results

### Static-Residualized kNN

```text
static_residualized_probefit:
  adjusted mean: +0.0444
  adjusted min:  +0.0220
  positive:      9/9

static_residualized_shuffled_probefit:
  adjusted mean: +0.0005
  adjusted min:  -0.0131
  positive:      5/9
```

Diagnostic residualizers were weaker:

```text
crossfit residualized action-effect: mean +0.0071
global residualized action-effect:   mean +0.0034
```

This means the positive result is strongest under the preregistered probe-fit
residualizer.  It should be read as evidence for additional structure, not as a
standalone method gate.

Residual energy was not near zero:

```text
probe-fit residual norm fraction:
  rgb:   mean 1.64
  range: mean 1.67
```

The residualizer therefore did not collapse the features.  The fractions above
one also mean the probe-fit static predictor transfers poorly to held-out
action columns; this is a diagnostic caveat.

### Static-Conditioned kNN

Action-effect remains positive inside all static-similarity bins:

```text
highest:  +0.0618, positive 9/9
high_mid: +0.0584, positive 9/9
low_mid:  +0.0559, positive 9/9
lowest:   +0.0576, positive 9/9
```

Shuffled controls remain near null:

```text
highest:  -0.0023
high_mid: +0.0035
low_mid:  -0.0089
lowest:   +0.0017
```

### Literature Metrics

Lightweight calibration with 10 permutations:

```text
cycle-kNN, probe-fit residualized:
  calibrated mean +0.0123
  positive 8/9

cycle-kNN, shuffled probe-fit residualized:
  calibrated mean -0.0039

CKNNA, probe-fit residualized:
  calibrated mean +0.0329
  positive 9/9

CKNNA, shuffled probe-fit residualized:
  calibrated mean -0.0005
```

## Decision

The static-control diagnostic is **supportive but not sufficient**.

Supported:

```text
The real-Powderworld action-effect signal is not merely identical to the
original static kNN signal.  It survives probe-fit static residualization and
static-conditioned candidate sets.
```

Still not supported:

```text
action-effect is a better convergent object than static representation
```

Reason:

```text
the original unconditioned real-Powderworld B6 static alignment is still much
stronger than action-effect alignment.
```

Therefore:

```text
AERA implementation remains blocked.
Path-Building / Sand-Pushing audit remains premature.
The next experiment is preregistered Stage B6R, not AERA and not task audit.
```

Stage B6R is preregistered in:

```text
docs/stageb6r_preregistration.md
```

## Output Files

```text
outputs/stageb_b6_realpw_v1_cpu/static_controls_primary_lit/stageb6_static_control_summary.csv
outputs/stageb_b6_realpw_v1_cpu/static_controls_primary_lit/stageb6_static_control_summary.json
outputs/stageb_b6_realpw_v1_cpu/static_controls_primary_lit/stageb6_static_residualized_knn.csv
outputs/stageb_b6_realpw_v1_cpu/static_controls_primary_lit/stageb6_static_conditioned_knn.csv
outputs/stageb_b6_realpw_v1_cpu/static_controls_primary_lit/stageb6_static_control_literature_metrics.csv
```

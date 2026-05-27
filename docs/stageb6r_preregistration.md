# Stage B6R Preregistration

Date: 2026-05-27 JST

## Purpose

Stage B6R is a confirmatory real-Powderworld static-control replication.  It is
not AERA, not Stage C, not PSP/Dreamer, and not a task audit.

The discovery set is real-Powderworld B6 v1 plus the posthoc static-control
diagnostic.  Those results motivate this preregistered replication but do not
count as confirmatory evidence.

## Revised Thesis

```text
Static alignment can capture shared scene layout.
Action-conditioned effect alignment should capture additional
intervention-specific cross-modal structure that survives static controls.
```

The old stronger thesis is not tested here:

```text
action-effect is the better convergent object than static representation
```

Real Powderworld B6 v1 already showed raw static alignment is stronger than raw
action-effect alignment.  B6R asks whether action-effect alignment contains
extra non-static structure.

## Fixed Experimental Design

Output root:

```text
outputs/stageb6r_realpw_v1_cpu
```

Data generation:

```text
backend: powderworld
data seeds: 10, 11, 12
n_states: 128
k_actions: 32
grid_size: 64
horizon: 4
channels: rgb, range, noisy_rgb, gray_rgb, blur_rgb
dense sampling: full-states
```

Analysis:

```text
split seeds: 101, 103, 107
PCA dim: 32
representation: pca_probe_only
normalization: probe_action_type_apply
k: 10
jitter: 0
target pair: rgb:range
max analyzed states: 128
expected runs: 9
```

The B6 diagnostic analysis still reports the original unconditioned rows, but
the B6R primary decision uses static-control outputs.

## Primary Static Controls

Primary residualized row:

```text
static_residualized_probefit
```

Definition:

```text
fit same-channel static-to-action-effect ridge residualizer on probe action
columns and apply it to held-out action columns
```

Primary null:

```text
static_residualized_shuffled_probefit
```

Secondary static-conditioned rows:

```text
action_effect_static_conditioned
shuffled_static_conditioned
```

The four preregistered bins are:

```text
lowest
low_mid
high_mid
highest
```

Each bin uses query-local chance adjustment because candidate pools differ by
query and bin.

Diagnostic-only residualizers:

```text
static_residualized_global
static_residualized_crossfit
```

They cannot rescue a failed probe-fit primary row.

## Literature Metrics

Run lightweight calibrated local-graph diagnostics on the residualized features:

```text
cycle-kNN
CKNNA
permutation repeats: 10
```

CKNNA is the primary literature-style diagnostic for the B6R gate.  cycle-kNN
is reported as secondary.

## Go Criteria

B6R passes only if all of the following hold:

```text
static_residualized_probefit adjusted mean >= +0.03
static_residualized_probefit positive in 9/9 runs
static_residualized_shuffled_probefit mean <= +0.01
static_residualized_probefit minus shuffled mean >= +0.03
static_residualized_probefit CKNNA mean >= +0.02
static_residualized_probefit CKNNA positive in 9/9 runs
action_effect_static_conditioned adjusted mean > 0 in all four bins
residual norm fraction is not near zero in rgb or range
all 9 runs are present
```

If the original unconditioned static row remains stronger than action-effect,
that does not fail B6R.  It only keeps the claim at the D-prime level rather
than the old Framing-D level.

## No-Go Criteria

Stop the AERA route if any of the following occurs:

```text
required runs are missing
residualized action-effect adjusted mean < +0.03
residualized action-effect is not positive in all runs
residualized shuffled reaches similar magnitude
residualized CKNNA mean < +0.02
residualized CKNNA is not positive in all runs
static-conditioned alignment disappears in multiple bins
residual energy is near zero in either channel
```

## Forbidden Interpretations

Do not claim:

```text
AERA is validated
action-effect is better than static representation
static alignment is irrelevant
scalar observability weighting is revived
Path-Building or Sand-Pushing audit is now authorized
```

B6R can only decide whether the real-Powderworld action-effect signal has
additional non-static cross-modal structure.

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
TARGET_PAIRS='rgb:range' \
bash scripts/submit_stageb_b6_realpw_analysis_array.sh
```

Static controls:

```bash
PYTHONPATH=src bash scripts/run_stageb6r_static_controls.sh \
  outputs/stageb6r_realpw_v1_cpu
```

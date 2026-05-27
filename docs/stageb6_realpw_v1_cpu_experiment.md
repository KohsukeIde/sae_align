# Stage B.6 Real Powderworld V1 CPU Experiment

Date: 2026-05-26

## Purpose

This experiment checks the Framing-D prerequisite:

> Before implementing AERA, verify that B6-style action-effect alignment is not
> only a ToyPowderWorld artifact.

This is an alignment diagnostic, not a behavioral detector, not PSP/Dreamer, and
not an AERA prototype.

## Precommit Link

The relevant preregistered decision rules are in
`docs/framing_d_aera_precommit.md`.

Main gate:

- action-effect alignment must exceed static and shuffled controls
- primary adjusted effect should be at least `+0.05` for a clear pass
- `+0.03` to `+0.05` is partial
- `< +0.03` is fail / no-go

The Path-Building / Sand-Pushing audit is not run before this gate. Its role is
determined by this real-Powderworld B6 result.

## Implementation Notes

Real Powderworld data is generated with:

```text
scripts/make_staged0_powderworld_dataset.py --backend powderworld
```

The generator was patched to emit Stage-B-compatible fields:

```text
channels
static_obs_sample_indices
action_array_schema
obs0_<channel>
delta_<channel>
detect_<channel>
```

`make_balanced_action_split.py` was patched to read the real Powderworld action
schema:

```text
["element", "x", "y", "radius"]
```

Generation and analysis are split so each real Powderworld dataset is generated
once per data seed and reused across split/PCA cells.

## Commands

Local smoke:

```bash
PYTHONPATH=src:. bash scripts/run_stageb_b6_realpw_smoke.sh \
  outputs/stageb_b6_realpw_smoke
```

V1 generation:

```bash
OUT=outputs/stageb_b6_realpw_v1_cpu \
DATA_SEEDS='0 1 2' \
N_STATES=128 \
K_ACTIONS=32 \
GRID_SIZE=64 \
HORIZON=4 \
CHANNELS='rgb range noisy_rgb gray_rgb blur_rgb' \
WALLTIME=04:00:00 \
bash scripts/submit_stageb_b6_realpw_generate_array.sh
```

V1 analysis:

```bash
OUT=outputs/stageb_b6_realpw_v1_cpu \
DATA_SEEDS='0 1 2' \
SPLIT_SEEDS='17 29 43' \
PCA_DIMS='16 32' \
WALLTIME=04:00:00 \
bash scripts/submit_stageb_b6_realpw_analysis_array.sh
```

Aggregation:

```bash
PYTHONPATH=src python scripts/summarize_stageb6_grid.py \
  --root outputs/stageb_b6_realpw_v1_cpu \
  --out outputs/stageb_b6_realpw_v1_cpu \
  --expected-report-dirs 18
```

## Scale

Generation:

```text
data seeds: 0, 1, 2
n_states: 128
k_actions: 32
grid_size: 64
horizon: 4
channels: rgb, range, noisy_rgb, gray_rgb, blur_rgb
dense rows per seed: 4096
```

Analysis:

```text
split seeds: 17, 29, 43
PCA dims: 16, 32
normalization: probe_action_type_apply
representation: pca_probe_only
k: 5, 10, 20
jitter: 0, 1e-5, 1e-4
target pair: rgb:range
analysis cells: 18
```

## Smoke Result

The local smoke completed end-to-end:

```text
outputs/stageb_b6_realpw_smoke/stageb6_grid_summary.json
```

This verified Powderworld import, dataset schema, balanced split, PCA encoder
training, B6 analyzer, and grid aggregation.

## V1 Result

Primary cell:

```text
pca_probe_only / probe_action_type_apply / d=32 / k=10 / jitter=0
```

Summary:

```text
n_runs: 9
rgb-range adjusted mean: +0.0339
rgb-range adjusted min:  +0.0127
rgb-range positive: 9/9
redundancy positive: 27/27
AE > static CI: 0/9
AE > shuffled CI: 8/9
detect_geom Spearman mean: +0.2619
```

Absolute primary-cell comparison:

```text
action-effect heldout: mean +0.0339, positive 9/9
static heldout:        mean +0.1179, positive 9/9
shuffled heldout:      mean +0.0031
```

K sweep at jitter 0:

```text
k=5:  mean +0.0273, positive 9/9
k=10: mean +0.0339, positive 9/9
k=20: mean +0.0473, positive 9/9
```

Literature-metric diagnostics at d=32:

```text
cycle-kNN:
  action-effect mean +0.0138, positive 9/9, p<=0.05 in 2/9
  static        mean +0.1238, positive 9/9, p<=0.05 in 9/9

CKNNA:
  action-effect mean +0.0190, positive 8/9, p<=0.05 in 5/9
  static        mean +0.1212, positive 9/9, p<=0.05 in 9/9
```

## Decision

This is **partial positive but not a Framing-D pass**.

Positive:

- real Powderworld has a nonzero action-effect rgb-range signal
- the signal is positive in all primary-cell runs
- redundancy controls pass
- shuffled controls are much weaker
- continuous detectability still correlates positively with overlap

Negative:

- the primary effect is only `+0.0339`, below the clear-pass threshold `+0.05`
- static rgb-range alignment is much stronger than action-effect alignment
- `AE > static CI` is `0/9`
- cycle-kNN and CKNNA also favor static over action-effect

Therefore:

```text
Do not implement AERA yet.
Do not build a custom simulator.
Do not claim that action-effect representation is the convergent object on real
Powderworld.
```

Under the precommitted audit rules, this result is at best the **partial** branch:

```text
effect size +0.03 to +0.05, but central static comparison fails
```

If the survey still favors Framing D, the next experiment is not AERA-v1. It is
the precommitted Path-Building / Sand-Pushing audit for selecting an
observation-critical AERA evaluation environment. That audit should be
precommitted separately because the current task-data script is DestroyAll
specific.

## Static-Control Addendum

After this result, a posthoc static-control diagnostic was run and documented in:

```text
docs/stageb6_static_control_postmortem.md
```

It found that action-effect signal survives probe-fit static residualization and
static-conditioned kNN, but this does not change the B6 v1 decision:

```text
static remains stronger in the original unconditioned real-Powderworld metric.
AERA implementation remains blocked.
```

## Output Files

```text
outputs/stageb_b6_realpw_v1_cpu/stageb6_grid_summary.json
outputs/stageb_b6_realpw_v1_cpu/stageb6_primary_cell_summary.csv
outputs/stageb_b6_realpw_v1_cpu/stageb6_decision_summary.csv
outputs/stageb_b6_realpw_v1_cpu/stageb6_literature_metrics.csv
```

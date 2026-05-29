# Static-Weak Setup Exploration

Date: 2026-05-29 JST

## Purpose

This is an exploratory real-Powderworld analysis after Stage B6R ended as a
strict no-go / near-miss.  It addresses the setup-level loophole:

```text
rgb:range may be static-strong because both channels expose the same scene
layout and occupancy.
```

The goal is to test whether any existing real-Powderworld channel pair is less
static-dominated and therefore more favorable to action-effect alignment.

This phase is not AERA, not Stage C, not a task audit, and not custom simulator
work.  It is exploratory sensitivity mapping.  It cannot rescue B6R and cannot
authorize AERA implementation.  At most, it can nominate one frozen cell for a
later preregistered replication on new seeds.

## Channels

Use real Powderworld with:

```text
rgb
range
local
edge
noisy_rgb
gray_rgb
blur_rgb
```

`edge` is an observation-derived RGB edge diagnostic, not a primary redundancy
control and not clean cross-modal evidence.  It is generated from rendered RGB
gradients inside `scripts/make_staged0_powderworld_dataset.py`.  Any `edge`
result is diagnostic-only within this exploratory phase.

## Pairs

```text
rgb:range
rgb:edge
rgb:local
range:local
rgb:noisy_rgb
rgb:gray_rgb
rgb:blur_rgb
```

Roles:

```text
rgb:range           static-strong reference pair
rgb:edge            derived diagnostic pair
rgb:local           action-site diagnostic pair
range:local         geometry-vs-local diagnostic pair
rgb:noisy/gray/blur redundancy controls
```

## Fixed Exploratory Grid

Output root:

```text
outputs/static_weak_realpw_v2_cpu
```

Data:

```text
data seeds: 20, 21, 22
n_states: 128
k_actions: 32
grid_size: 64
horizon: 4
dense sampling: full-states
```

Analysis:

```text
split seeds: 201, 203, 207
PCA dim: 32
normalization: probe_action_type_apply
representation: pca_probe_only
k: 10
jitter: 0
expected runs: 9
```

## Primary Report

For each pair, report:

```text
static_adjusted
action_effect_adjusted
shuffled_adjusted
residualized_action_effect
residualized_shuffled
action_minus_static
action_minus_shuffled
residualized_minus_shuffled
CKNNA_action
CKNNA_static
CKNNA_residualized
```

## Interpretation Rules

This is exploratory, so it cannot directly open AERA implementation.

A candidate-generating cell must satisfy all of:

```text
all planned runs present
raw static no longer strongly dominates action-effect
static_residualized_probefit adjusted mean >= +0.03
static_residualized_probefit minus shuffled mean >= +0.03
static_residualized_shuffled_probefit mean <= +0.01
residualized CKNNA mean >= +0.02 and positive in all runs
static-conditioned action-effect adjusted mean > 0 and > shuffled in every bin
residual_norm_fraction min >= 0.10 in both channels
```

Passing this exploratory screen still only permits a fresh confirmatory
preregistration.  It does not permit AERA implementation.

If a non-redundancy diagnostic pair shows:

```text
action_effect > static
and action_effect > shuffled
```

then the stronger Framing D remains setup-dependent but viable.

If:

```text
action_effect ~= static
and residualized_action_effect > shuffled
```

then Framing D-prime remains the honest working thesis.

If:

```text
action_effect ~= shuffled
```

across all non-redundancy diagnostic pairs, stop the AERA route.

If redundancy controls show the same residualized behavior as the diagnostic
pairs, treat the pattern as likely metric artifact.

Forbidden during this phase:

```text
new tasks
Path-Building / Sand-Pushing generation
custom simulators
task audits
C0/D0 detector variants
Action-IV Step 3
PSP / Dreamer / RL comparisons
new observation channels beyond the listed real-Powderworld render/derived set
threshold lowering after seeing outputs
excluding failed cells
changing residualizer/bin/k/PCA settings without labeling the result exploratory
```

## Commands

Generation:

```bash
OUT=outputs/static_weak_realpw_v2_cpu \
DATA_SEEDS='20 21 22' \
N_STATES=128 \
K_ACTIONS=32 \
GRID_SIZE=64 \
HORIZON=4 \
CHANNELS='rgb range local edge noisy_rgb gray_rgb blur_rgb' \
bash scripts/submit_stageb_b6_realpw_generate_array.sh
```

Analysis:

```bash
OUT=outputs/static_weak_realpw_v2_cpu \
DATA_SEEDS='20 21 22' \
SPLIT_SEEDS='201 203 207' \
PCA_DIMS='32' \
K_VALUES='10' \
JITTER_EPSILONS='0' \
JITTER_SEEDS='0' \
NORMALIZATION_MODES='probe_action_type_apply' \
REPRESENTATIONS='pca_probe_only' \
CHANNELS='rgb range local edge noisy_rgb gray_rgb blur_rgb' \
TARGET_PAIRS='rgb:range rgb:edge rgb:local range:local rgb:noisy_rgb rgb:gray_rgb rgb:blur_rgb' \
bash scripts/submit_stageb_b6_realpw_analysis_array.sh
```

Static controls:

```bash
PYTHONPATH=src bash scripts/run_static_weak_static_controls.sh \
  outputs/static_weak_realpw_v2_cpu
```

Pair summary:

```bash
PYTHONPATH=src python scripts/summarize_static_weak_pairs.py \
  --root outputs/static_weak_realpw_v2_cpu
```

Until this summary is written, use the raw CSVs rather than
`stageb6_static_control_summary.csv`, because the original static-control
summary was designed around a single `rgb:range` primary pair.

## v1 Split Loophole

An initial `outputs/static_weak_realpw_v1_cpu` attempt used the old real
Powderworld action-bank sampler, which chose uniformly over material elements.
Because `empty` is only one element, 32-action banks contained too few erase
actions and some probe/held-out splits had action types present only on one
side. That makes `probe_action_type_apply` ill-defined and is a split-design
failure, not a scientific result.

The real Powderworld adapter now samples erase/place intervention families
explicitly, with erase probability 0.25, so both probe and held-out splits can
contain both action families. The exploratory run is therefore restarted as
`v2`.

# Stage C0 v1 CPU Experiment

Date: 2026-05-19

## Purpose

Stage C0 is a behavioral smoke test, not a PSP/Dreamer comparison. It asks
whether the Stage B.6 continuous action-effect observability signal can be used
as a training-weight signal for lightweight prediction heads.

This run uses the existing Stage B.6-style ToyPowderWorld datasets with static
observations and dense deltas.

## Implementation Hardening

Before running the final v1 grid, the patch was tightened to address leakage and
metric loopholes:

- observability ranks are fit inside each train split, not globally over
  train/val/test rows;
- the primary observability mix is geometric detectability,
  `sqrt(rank_train(detect_rgb) * rank_train(detect_range))`;
- `event_response`, `semantic`, and `edge` are rejected as input or
  observability channels;
- `delta_event_response` is required for primary runs; `world_delta` fallback is
  diagnostic-only;
- event and changed-cell F1 are scored from predicted magnitudes, not signed raw
  deltas;
- summaries include pairwise gates against `uniform`, `change_mask`,
  `shuffled_observability`, and the best non-oracle baseline.

## Run

Smoke:

```bash
PYTHONPATH=src bash scripts/run_stagec0_smoke.sh outputs/stagec0_smoke_patch_v2
```

V1 array runs:

```bash
DATA=outputs/stageb_b6_primary_cell_v1_cpu/seed_0/split_17/pca_32/stage0/stage0_dataset.npz \
OUT=outputs/stagec0_v1_cpu_array_b6seed0_v3 \
bash scripts/submit_stagec0_v1_cpu_array.sh

DATA=outputs/stageb_b6_primary_cell_v1_cpu/seed_1/split_17/pca_32/stage0/stage0_dataset.npz \
OUT=outputs/stagec0_v1_cpu_array_b6seed1_v3 \
bash scripts/submit_stagec0_v1_cpu_array.sh

DATA=outputs/stageb_b6_primary_cell_v1_cpu/seed_2/split_17/pca_32/stage0/stage0_dataset.npz \
OUT=outputs/stagec0_v1_cpu_array_b6seed2_v3 \
bash scripts/submit_stagec0_v1_cpu_array.sh
```

Jobs:

```text
1777234[].pbs1
1777252[].pbs1
1777253[].pbs1
```

Scale:

- data seeds: `0 1 2`
- per data seed: `8` cells = `4` observability mixes x `2` input sets
- input sets: `rgb`, `rgb range`
- observability mixes: `geom`, `mean`, `product`, `min`
- methods: `uniform`, `change_mask`, `observability`,
  `shuffled_observability`, `oracle_event`
- split seeds inside each cell: `0 1 2 3 4`
- dataset per data seed: `8192` dense state-action rows

Runtime:

- wall-clock per qsub task: about `12s` to `21s`
- memory per qsub task: about `1.6GB` to `1.8GB`
- exit status: `0` for all final v1 tasks

Aggregate output:

```text
outputs/stagec0_v1_cpu_array_b6seeds012_v3/
  stagec0_grid_summary.csv
  stagec0_method_delta_summary.csv
  stagec0_pairwise_decision_summary.csv
  stagec0_pairwise_decision_aggregate.csv
  stagec0_grid_summary.json
```

## Main Result

Stage C0 v1 is a No-go.

Across all `24` data-seed/cell combinations, observability weighting never beat
uniform on event F1 or OOD event F1:

```text
go_event_or_ood_vs_uniform_count: 0 for every input/mix cell
go_event_or_ood_exceeds_stageb_ref_count: 0 for every input/mix cell
go_beats_best_nonoracle_event_count: 0 for every input/mix cell
```

The primary-like cell `input_rgb / obs_geom` was negative:

| metric | mean over 3 data seeds |
| --- | ---: |
| event F1 delta vs uniform | `-0.0408` |
| OOD event F1 delta vs uniform | `-0.0449` |
| event delta minus B6 ref | `-0.0908` |
| OOD delta minus B6 ref | `-0.0949` |
| observability minus change-mask event | `+0.0028` |
| observability minus shuffled event | `-0.0340` |
| observability minus best non-oracle event | `-0.0408` |

The best secondary input family, `rgb range`, was also negative:

| cell | event F1 delta vs uniform | OOD event F1 delta vs uniform |
| --- | ---: | ---: |
| `input_rgb_range / obs_geom` | `-0.0149` | `-0.0246` |
| `input_rgb_range / obs_min` | `-0.0147` | `-0.0220` |

## Interpretation

The Stage B.6 continuous-observability signal did not transfer into this
minimal weighted-ridge C0 objective. This blocks PSP/Dreamer comparisons.

The stronger caveat is that `oracle_event` weighting also failed to beat
uniform:

```text
oracle_event_minus_uniform_event_mean < 0 in every cell
```

This suggests the current C0 task/model/weighting setup may be a weak detector
of useful weighting, rather than proving that observability is useless. In
particular, weighted ridge on dense deltas may be dominated by uniform fitting,
threshold choice, or target imbalance.

## Decision

```text
Stage C0 v1: No-go.
Stage C1 PSP-like comparison: blocked.
Stage B.6 signal: still valid as weak alignment evidence, but not behaviorally
validated by this C0 implementation.
Next action: redesign C0 target/model/weighting or move to a slightly more
expressive neural smoke before PSP/Dreamer.
```

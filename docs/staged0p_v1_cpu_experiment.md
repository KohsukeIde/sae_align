# Stage D0' v1 CPU Experiment

Date: 2026-05-20

## Purpose

Stage D0' is the final ToyPowderWorld follow-up after Stage C0/C0.5 No-go. It
tests exactly one predictor-grounded observability score before either stopping
Toy Stage C or opening a separately preregistered environment/target/model
phase.

The precommitted primary score is:

```text
predictor_grounded =
  rank_train(entropy_uniform_event_predictor)
  * rank_train(obs_geom)
```

where `obs_geom` is the geometric mean of train-fitted `detect_rgb` and
`detect_range` ranks.

The oracle-event gate is hard. If `oracle_event` does not beat uniform by
`>= +0.10` on event AUPRC or event F1, predictor-grounded positives are
diagnostic-only.

## Implementation Hardening

Before the v1 run, D0' was patched to close the main loopholes found in review:

- predictor-grounded score now matches the precommit definition;
- uniform-predictor train uncertainty uses train-fold cross-fit predictions;
- AUPRC handles score ties as groups instead of row-order artifacts;
- decision summaries use AUPRC-or-F1 behavior deltas, not AUPRC only;
- aggregate pass checks use metric-specific best rows for AUPRC and F1 rather
  than reusing the best behavior row;
- oracle failure forces predictor-grounded to `interpretable=false`;
- local v1 fails on missing datasets and uses `--expected-report-dirs`;
- BLAS threads are pinned in runners;
- qsub helper is a site-local template using nonblank dataset counting and
  1-based PBS array IDs; it was not used for this local run;
- JSON summaries are strict and do not emit `NaN`.

## Run

Local CPU was sufficient. qsub/HC was not used.

```bash
rm -rf outputs/staged0p_v1_cpu_v3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=src:. bash scripts/run_staged0p_v1_cpu.sh outputs/staged0p_v1_cpu_v3
```

Default datasets:

```text
outputs/stageb_b6_primary_cell_v1_cpu/seed_0/split_17/pca_32/stage0/stage0_dataset.npz
outputs/stageb_b6_primary_cell_v1_cpu/seed_1/split_17/pca_32/stage0/stage0_dataset.npz
outputs/stageb_b6_primary_cell_v1_cpu/seed_2/split_17/pca_32/stage0/stage0_dataset.npz
```

Output:

```text
outputs/staged0p_v1_cpu_v3/
  run_0/reports/
  run_1/reports/
  run_2/reports/
  staged0p_grid_results.csv
  staged0p_grid_summary.csv
  staged0p_decision_grid.json
```

Runtime/resource:

- execution: local CPU only; qsub/HC was not used;
- wall-clock: completed locally without queueing; qsub/HC was unnecessary for
  this three-dataset v1 grid;
- BLAS threads: runners pin `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` by default;
- expected report dirs: `3`; found `3`;
- reruns should use a fresh output directory or clear the old one first.

## Result

Aggregate decision:

```text
decision_branch: branch_3_detector_failed_stop_toy
n_report_dirs: 3
n_rows: 615
```

Key deltas versus uniform:

| method | best behavior delta | best AUPRC delta | interpretation |
| --- | ---: | ---: | --- |
| `oracle_event` | `+0.0015` | `-0.0039` | fails hard oracle gate |
| `predictor_grounded` | `+0.0898` | `+0.0898` | raw positive but below `+0.10` and uninterpretable because oracle failed |
| `change_mask` | `+0.0712` | `+0.0712` | non-oracle baseline lift |
| `observability` | `+0.0545` | `+0.0545` | weak, near Stage-B reference scale |
| `shuffled_predictor_grounded` | `+0.0017` | `-0.0030` | control near null |
| `lossgrad_observability` | `+0.1203` | `+0.1203` | label-aware diagnostic only |

The predictor-grounded score is consistently positive:

```text
predictor_grounded best alpha: 16
test_auprc_delta_mean: +0.0898
test_auprc_delta_min:  +0.0470
positive fraction: 1.0
```

However, it does not pass D0':

```text
oracle_pass_plus_0p10: false
predictor_grounded_interpretable: false
predictor_grounded_pass_plus_0p10_and_controls: false
```

## Decision

```text
Stage D0' v1: No-go.
ToyPowderWorld Stage C: stop.
Stage C1 / PSP-like comparison: blocked.
```

D0' produced a raw predictor-grounded AUPRC lift, but the oracle-event detector
did not pass. Per the precommit, predictor-grounded results cannot be
interpreted as behavioral validation when oracle-event weighting itself fails.

The correct next step is not C0.7/C0.8 detector tweaking. Any continuation
requires a separately preregistered phase:

- environment migration with an oracle-positive detector audit first;
- target/model redesign with a new precommit;
- or project reframing around diagnostic alignment evidence rather than a
  ToyPowderWorld behavioral utility claim.

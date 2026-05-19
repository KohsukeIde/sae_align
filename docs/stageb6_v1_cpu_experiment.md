# Stage B.6 v1 CPU Experiment

Date: 2026-05-19

## Purpose

Stage B.6 tests whether the weak Stage B.5 `rgb-range` held-out
same-action-set signal is a PCA, normalization, k, tie, or measurement-primitive
artifact. This is a diagnostic hardening step, not Stage C.

## Run

The full grid was run as an ABCI CPU array:

```bash
DATA_SEEDS="0 1" SPLIT_SEEDS="17 29" PCA_DIMS="16 32 64 128" \
  OUT=outputs/stageb_b6_v1_cpu \
  PYTHONPATH=src bash scripts/submit_stageb_b6_v1_cpu_array.sh

PYTHONPATH=src python scripts/summarize_stageb6_grid.py \
  --root outputs/stageb_b6_v1_cpu \
  --out outputs/stageb_b6_v1_cpu
```

Scale:

- array tasks: `16`
- data seeds: `0 1`
- split seeds: `17 29`
- PCA components: `16 32 64 128`
- generated states/actions per task: `256 x 32`
- dense delta rows per task: `4096`
- analyzed complete states: `128`
- representations: `pca_probe_only`, `pca_all_action`, `raw_delta`,
  `random_projection`
- normalizations: `none`, `probe_global_apply`, `probe_action_type_apply`
- k values: `5 10 20`
- jitter epsilons: `0, 1e-6, 1e-5, 1e-4, 1e-3`
- jitter seeds for nonzero epsilons: `0..9`
- permutation repeats: `200`
- bootstrap repeats: `200`

Runtime:

- wall-clock per task: about `1h40m` to `1h48m`
- CPU time per task: about `42h` to `45h`
- exit status: `0` for all 16 tasks

Aggregate reports:

- `outputs/stageb_b6_v1_cpu/stageb6_primary_cell_summary.csv`
- `outputs/stageb_b6_v1_cpu/stageb6_decision_summary.csv`
- `outputs/stageb_b6_v1_cpu/stageb6_paired_differences.csv`
- `outputs/stageb_b6_v1_cpu/stageb6_observability_score_correlation.csv`
- `outputs/stageb_b6_v1_cpu/stageb6_measurement_sanity.csv`
- `outputs/stageb_b6_v1_cpu/stageb6_grid_summary.json`

## Primary Cell Result

The preregistered primary replication cell was:

```text
pca_probe_only / probe_action_type_apply / d=32 / k=10 / jitter=0
```

Result:

| metric | value |
| --- | ---: |
| runs | `4` |
| `rgb-range` adjusted mean | `+0.0437` |
| `rgb-range` adjusted min | `+0.0150` |
| `rgb-range` positive runs | `4/4` |
| redundancy positive rows | `12/12` |
| AE > static paired CI | `3/4` |
| AE > shuffled paired CI | `3/4` |
| `detect_geom_rank_mean` Spearman | `+0.318` |

This reproduces the Stage B.5 magnitude rather than amplifying it.

## Robustness

For `pca_probe_only / probe_action_type_apply`:

| components | adjusted mean | adjusted min | positive fraction | AE > static CI fraction | AE > shuffled CI fraction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `16` | `+0.0455` | `+0.0041` | `1.000` | `0.669` | `0.669` |
| `32` | `+0.0408` | `-0.0112` | `0.935` | `0.707` | `0.744` |
| `64` | `+0.0272` | `+0.0028` | `1.000` | `0.384` | `0.343` |
| `128` | `+0.0215` | `-0.0034` | `0.976` | `0.463` | `0.220` |

Diagnostic representations:

| representation | adjusted mean | positive fraction |
| --- | ---: | ---: |
| `random_projection / probe_action_type_apply` | `+0.0301` | `1.000` |
| `raw_delta / probe_action_type_apply` | `+0.0180` | `0.917` |

The signal is not restricted to `d=32`, but it is stronger in PCA-compressed
features than in raw deltas.

## Measurement Sanity

Calibrated CKA/RSA/ridge diagnostics were added because PRH-style kNN evidence
can be fragile and raw similarity scores can be misleading. The B.6 sanity
checks do not provide a single clean replacement metric, but they support the
same qualitative reading:

- action-conditioned RSA is consistently positive for `rgb-range`;
- state-flat RSA/CKA generally point in the positive direction under PCA;
- ridge transfer is numerically unstable in some high-dimensional conditions
  and should not be used as primary evidence in this pilot;
- shuffled controls can still be high for some global metrics, so calibrated
  CKA/RSA are diagnostics rather than final claims.

The literature-method notes behind these choices are in
`docs/alignment_metric_notes.md`. They explicitly record that the three checked
papers use different alignment primitives: raw mutual kNN and gallery-density
stress tests, permutation-calibrated metric families including mKNN/cycle-kNN/
CKNNA/CKA/RSA, and CCA/subspace-plus-retrieval alignment for 3D-text features.
Stage B.6 v1 covered only the subset implemented at that time. After B.6 v1,
diagnostic-only cycle-kNN, CKNNA, CCA, and SVCCA rows were added in a separate
`b6_literature_metrics.csv` output. They are not part of the B.6 v1 primary
summary.

For the primary replication cell
`pca_probe_only / probe_action_type_apply / d=32`, `rgb-range` measurement
sanity over the four data/split runs was:

| control | measurement | calibrated mean | calibrated min | p<=0.05 runs | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| action-effect | action-conditioned RSA | `+0.293` | `+0.231` | `4/4` | stable positive |
| action-effect | state-flat RSA | `+0.169` | `+0.136` | `4/4` | stable positive |
| action-effect | state-flat linear CKA | `+0.049` | `+0.016` | `2/4` | positive but weaker |
| action-effect | ridge R2 | `+4.162` | `-592.555` | `0/4` | numerically unstable |
| static | state-flat RSA | `+0.025` | `+0.009` | `2/4` | weak calibrated static structure |
| shuffled action columns | state-flat RSA | `+0.043` | `+0.001` | `2/4` | shuffled controls can retain global structure |

Thus the measurement sanity checks support a weak positive action-effect
direction, but ridge transfer and global metrics are not clean enough to replace
the kNN result.

## Review Method

A read-only literature-method audit was run before this doc update. The audit
was intentionally fixed-scope: inspect the three specified papers and current
repo code/docs, then report method differences and loopholes. It was not used
as an open-ended search engine.

The audit confirmed that the papers use different primitives:

- `2604.18572`: L2-normalized mutual kNN overlap over shared galleries, with
  stress tests for gallery size, `k`, and many-to-many correspondences.
- `2602.14486`: calibrated metric suite including mKNN, cycle-kNN, CKNNA,
  CKA/RSA/CCA-family metrics, with permutation-null correction.
- `2503.05283`: CCA-selected low-dimensional subspaces followed by affine or
  local CKA matching/retrieval.

B.6 implements the first two only partially and treats the third as motivation
for subspace diagnostics. The missing methods are recorded in
`docs/alignment_metric_notes.md`.

## Interpretation

Stage B.6 v1 is best labeled:

```text
partial robust positive
```

The B.5 `rgb-range` signal is not just a `k=10 / d=32` accident:

- the primary cell is positive in all four seed/split runs;
- redundancy controls remain positive;
- the signal mostly survives k, jitter, and PCA component sweeps;
- continuous observability remains positively associated with overlap.

However, this is still not a full pass:

- effect size remains small, around `+0.03` to `+0.05` in the best primary
  settings;
- paired CI support is incomplete;
- raw-delta and random-projection diagnostics are weaker;
- PCA/subspace denoising appears important;
- binary regular/blind strata remain weaker than continuous observability.

## Decision

```text
Stage B.6 v1: partial robust positive.
Stage C: not automatic, but no longer blocked by an obviously broken metric.
Binary strata framing: weak.
Continuous action-effect observability framing: strengthened.
Option 3 universal-coupler pivot: still premature.
```

Next work should use this result to design a smaller Stage C/proxy smoke only
if the objective is explicitly tied to continuous observability rather than
binary regular/blind strata.

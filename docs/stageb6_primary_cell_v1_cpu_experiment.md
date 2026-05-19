# Stage B.6 Primary-Cell Larger CPU Experiment

Date: 2026-05-19

## Purpose

This run expands the Stage B.6 primary-cell replication without moving to
Stage C. It checks whether the B.6 v1 `rgb-range` signal remains positive over
more data/split seeds and 256 complete states, while recording literature-
derived metrics as diagnostic-only outputs.

## Run

Submitted ABCI CPU array:

```bash
bash scripts/submit_stageb_b6_primary_cell_v1_cpu_array.sh

PYTHONPATH=src python scripts/summarize_stageb6_grid.py \
  --root outputs/stageb_b6_primary_cell_v1_cpu \
  --out outputs/stageb_b6_primary_cell_v1_cpu \
  --expected-report-dirs 18
```

Job:

```text
1777222[].pbs1
```

Scale:

- array tasks: `18`
- data seeds: `0 1 2`
- split seeds: `17 29 43`
- PCA dims: `16 32`
- generated state-action samples per task: `256 x 32 = 8192`
- dense delta rows per task: `8192`
- analyzed complete states: `256`
- representation: `pca_probe_only`
- normalization: `probe_action_type_apply`
- k values: `5 10 20`
- jitter epsilons: `0 1e-5 1e-4`
- literature metrics: cycle-kNN, CKNNA, CCA, SVCCA

Runtime:

- wall-clock per task: about `3m22s` to `4m43s`
- CPU time per task: about `26m50s` to `47m00s`
- exit status: `0` for all 18 tasks

Aggregate outputs:

- `outputs/stageb_b6_primary_cell_v1_cpu/stageb6_primary_cell_summary.csv`
- `outputs/stageb_b6_primary_cell_v1_cpu/stageb6_decision_summary.csv`
- `outputs/stageb_b6_primary_cell_v1_cpu/stageb6_literature_metrics.csv`
- `outputs/stageb_b6_primary_cell_v1_cpu/stageb6_grid_summary.json`

## Primary Cell

The fixed primary cell remains:

```text
pca_probe_only / probe_action_type_apply / d=32 / k=10 / jitter=0
```

Result:

| metric | value |
| --- | ---: |
| runs | `9` |
| `rgb-range` adjusted mean | `+0.0275` |
| `rgb-range` adjusted min | `+0.0112` |
| `rgb-range` positive runs | `9/9` |
| redundancy positive rows | `27/27` |
| AE > static paired CI | `9/9` |
| AE > shuffled paired CI | `8/9` |
| `detect_geom_rank_mean` Spearman | `+0.406` |
| non-ridge sanity calibrated mean | `+0.096` |

The effect size is smaller than B.6 v1 (`+0.0437`) but more stable across
seeds/splits. This supports the reading that the signal is weak but not a
single-seed artifact.

## Robustness Summary

From `stageb6_decision_summary.csv`:

| representation | components | adjusted mean | adjusted min | positive fraction | AE > static CI fraction | AE > shuffled CI fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pca_probe_only` | `16` | `+0.0296` | `+0.0053` | `1.000` | `0.885` | `0.840` |
| `pca_probe_only` | `32` | `+0.0268` | `-0.0001` | `0.996` | `0.912` | `0.838` |

## Literature Metrics

The literature metrics are diagnostic-only and are written to
`stageb6_literature_metrics.csv`. They are not mixed into the primary-cell
summary.

For `rgb-range / pca_probe_only / probe_action_type_apply`:

| control | metric | calibrated mean | calibrated min | p<=0.05 rows | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| action-effect | CKNNA | `+0.0334` | `+0.0137` | `18/18` | clean local-graph support |
| action-effect | cycle-kNN | `+0.0272` | `-0.0083` | `15/18` | mostly positive, stricter than kNN |
| action-effect | SVCCA | `+0.0486` | `+0.0054` | `14/18` | positive but subspace diagnostic |
| action-effect | CCA | `+0.0283` | `-0.0318` | `4/18` | weak / not primary |
| static | CKNNA | `-0.0017` | `-0.0083` | `0/18` | no local-graph static support |
| shuffled | CKNNA | `+0.0007` | `-0.0107` | `0/18` | shuffled graph control behaves |
| shuffled | SVCCA | `+0.0363` | `-0.0113` | `8/18` | SVCCA can retain global/subspace structure |

Interpretation:

- CKNNA is the cleanest literature-derived support: action-effect is positive,
  while static and shuffled graph controls are near null.
- cycle-kNN is mostly positive but stricter and less stable.
- CCA/SVCCA should remain diagnostic-only because subspace metrics can stay
  positive under controls and can be circular if promoted without a separate
  fit/eval split.

## Decision

```text
Stage B.6 larger primary-cell replication: positive but weak.
Stage C0 smoke: justified as a separate behavioral validation.
Full Stage C / PSP-like comparison: still premature.
Binary strata framing: weak.
Continuous action-effect observability framing: strengthened.
Literature metrics: diagnostic support, not primary gate replacement.
```

The next step should not be more strategy discussion. It should be either the
separate Stage C0 smoke or, if remaining within Stage B, a preregistered
diagnostic that promotes one literature metric only after defining its pass
criteria in advance.

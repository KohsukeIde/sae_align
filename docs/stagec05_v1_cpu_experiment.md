# Stage C0.5 v1 CPU Experiment

Date: 2026-05-19

## Purpose

Stage C0 v1 was No-go, and oracle-event weighting did not beat uniform. Stage
C0.5 tests whether this was simply because weighted ridge was a weak detector.

The precommitted gate is recorded in `docs/stagec05_precommit.md`:

- first establish an oracle-positive detector;
- use AUPRC/AUROC/balanced accuracy in addition to F1;
- require at least `+0.10` improvement before Stage C1 / PSP-like comparison;
- stop ToyPowderWorld Stage C if oracle-positive classifier detection fails.

## Implementation

Added `scripts/train_stagec05_detector.py`.

Detector:

- binary logistic classifier, NumPy full-batch training;
- inputs: `obs0_rgb` random projection plus action features;
- targets:
  - `event_present`;
  - `changed_any` diagnostic target;
- methods:
  - `uniform`;
  - `change_mask`;
  - `observability`;
  - `shuffled_observability`;
  - `oracle_event_class_weight`;
  - `oracle_event_sample_weight`;
  - `oracle_changed_class_weight`;
  - `oracle_changed_sample_weight`;
- alpha sweep: `0 2 4 8 16 32`;
- metrics: F1, AUPRC, AUROC, balanced accuracy, precision, recall, OOD variants,
  prevalence, ESS, top-decile weight mass, positive/negative event weight mass.

Also hardened C0 utilities:

- `percentile_rank` is tie-aware;
- binary oracle weights preserve binary ties instead of imposing artificial
  within-class ranks.

## Run

Event-present detector:

```bash
for SEED in 0 1 2; do
  PYTHONPATH=src:. python scripts/train_stagec05_detector.py \
    --data outputs/stageb_b6_primary_cell_v1_cpu/seed_${SEED}/split_17/pca_32/stage0/stage0_dataset.npz \
    --out outputs/stagec05_v1_cpu_b6seed${SEED} \
    --input-channels rgb \
    --methods uniform change_mask observability shuffled_observability \
      oracle_event_class_weight oracle_event_sample_weight \
      oracle_changed_class_weight oracle_changed_sample_weight \
    --alphas 0 2 4 8 16 32 \
    --seeds 0 1 2 3 4 \
    --channel-dim 64 \
    --epochs 120 \
    --lr 0.2 \
    --l2 1e-4
done
```

Changed-any detector:

```bash
for SEED in 0 1 2; do
  PYTHONPATH=src:. python scripts/train_stagec05_detector.py \
    --data outputs/stageb_b6_primary_cell_v1_cpu/seed_${SEED}/split_17/pca_32/stage0/stage0_dataset.npz \
    --out outputs/stagec05_v1_changed_cpu_b6seed${SEED} \
    --target-kind changed_any \
    --input-channels rgb \
    --methods uniform change_mask observability shuffled_observability \
      oracle_event_class_weight oracle_event_sample_weight \
      oracle_changed_class_weight oracle_changed_sample_weight \
    --alphas 0 2 4 8 16 32 \
    --seeds 0 1 2 3 4 \
    --channel-dim 64 \
    --epochs 120 \
    --lr 0.2 \
    --l2 1e-4
done
```

Runtime was local CPU only, about `1.3-2.0` minutes per data seed. HC/qsub was
not needed.

Aggregates:

```text
outputs/stagec05_v1_event_cpu_b6seeds012/
outputs/stagec05_v1_changed_cpu_b6seeds012/
```

## Event-Present Result

Best non-uniform event-present row:

| method | alpha | event F1 delta | event AUPRC delta | OOD event F1 delta | OOD event AUPRC delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `oracle_changed_class_weight` | `2` | `-0.0255` | `+0.0845` | `-0.0379` | `+0.0634` |

Observability rows:

| method | alpha | event F1 delta | event AUPRC delta | OOD event F1 delta | OOD event AUPRC delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `observability` | `32` | `-0.0234` | `+0.0723` | `-0.0394` | `+0.0487` |
| `observability` | `16` | `-0.0223` | `+0.0718` | `-0.0345` | `+0.0493` |
| `observability` | `8` | `-0.0150` | `+0.0693` | `-0.0227` | `+0.0488` |

Controls:

- `shuffled_observability` did not show the AUPRC lift; its event AUPRC deltas
  were near or below zero.
- `change_mask` had similar event AUPRC lift to observability (`+0.0700` at
  alpha `32`), so observability is not yet clearly distinct.
- `oracle_event_class_weight` did not improve AUPRC; at alpha `32`, event AUPRC
  delta was `-0.0988` and OOD AUPRC delta was `-0.0797`.

Interpretation:

The classifier detects some weighting effects in threshold-free AUPRC, but the
effect does not reach the precommitted `+0.10` threshold. F1 is generally worse
than uniform. The oracle-event detector is not oracle-positive.

## Changed-Any Result

The changed-any target is already easy for the uniform classifier:

```text
best changed-any AUPRC delta vs uniform: about +0.0005
best changed-any F1 delta vs uniform: about +0.0017
```

This target is saturated and cannot validate useful weighting.

## Decision

```text
Stage C0.5 v1: No-go.
Oracle-event classifier: failed oracle-positive gate.
Observability: weak AUPRC lift, below +0.10 and not clearly better than change-mask.
Changed-any detector: saturated; not useful as a weighting detector.
Stage C1 / PSP-like comparison: blocked.
C0.6 predictor-grounded redesign: not triggered, because the oracle-event gate did not pass.
```

The correct next move is not another C0.6/C0.7 detector tweak. Per
`docs/stagec05_precommit.md`, ToyPowderWorld Stage C should stop unless the
project explicitly chooses environment migration, target redesign, or a neural
detector phase as a new preregistered stage.

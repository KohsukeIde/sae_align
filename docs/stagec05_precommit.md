# Stage C0.5 Precommit: Oracle-Positive Detector Redesign

Date: 2026-05-19

## Status

Stage B.6 remains weak but stable alignment evidence. Stage C0 v1 is No-go:
observability weighting did not beat uniform on event/OOD F1, and oracle-event
weighting also failed. Therefore, Stage C1 / PSP-like comparison is blocked.

Stage C0.5 is not a method result. Its purpose is narrower:

> Build a detector in which an oracle event/change weighting signal can beat
> uniform. Only after that detector is oracle-positive can observability be
> interpreted.

## Fixed Questions

1. Can an oracle event/change weighting signal improve event prediction over
   uniform?
2. If oracle weighting works, does continuous observability also improve event
   prediction beyond uniform, change-mask, and shuffled observability?
3. If oracle works but observability fails, is exactly one
   predictor-grounded observability redesign worth trying?

## Required Detector Changes

Stage C0.5 must add a classifier-style detector and threshold-free metrics:

- event-present and event-type prediction with logistic or small-MLP heads;
- AUPRC, AUROC, balanced accuracy, precision, recall, and F1;
- event prevalence for train/val/test;
- alpha sweep for weights;
- effective sample size, top-decile mass, and positive/negative event weight
  mass;
- tie-aware percentile ranks for continuous scores;
- explicit class/sample weights for binary oracle scores.

Weighted ridge reconstruction alone is not an oracle-positive detector.

## Candidate Methods

Minimum methods:

```text
uniform
change_mask
observability
shuffled_observability
oracle_event_class_weight
oracle_event_sample_weight
oracle_changed_class_weight
oracle_changed_sample_weight
```

Optional diagnostic:

```text
oracle_event_target_weight
```

`oracle_*` methods are target-leakage diagnostics. They are allowed only to test
whether the detector can detect a useful weighting signal.

## Effect-Size Thresholds

Stage C1 can be considered only if a primary behavioral metric improves by at
least `+0.10` over uniform and also beats the relevant controls:

```text
event_f1_delta_vs_uniform >= +0.10
or event_f1_ood_delta_vs_uniform >= +0.10
or event_auprc_delta_vs_uniform >= +0.10
```

and:

```text
method > change_mask
method > shuffled_observability
```

Interpretation:

- `delta < +0.05`: No-go.
- `+0.05 <= delta < +0.10`: weak / inconclusive; do not proceed to PSP.
- `delta >= +0.10`: enough to justify Stage C1.
- `delta >= +0.15`: strong behavioral signal.

## Branches

### Branch 1: Best Case

Condition:

```text
oracle_event > uniform
observability > uniform
observability > change_mask
observability > shuffled_observability
observability improves event F1, OOD event F1, or event AUPRC by >= +0.10
```

Decision:

```text
Proceed to Stage C1 PSP-like comparison.
```

Stage C1 should include uniform, change-mask, PSP-like saliency,
observability, PSP-like + observability, and shuffled-observability controls.

### Branch 2: Median Case

Condition:

```text
oracle_event > uniform
observability <= uniform
or observability ~= shuffled_observability
```

Decision:

```text
Do not proceed to PSP.
Run exactly one predictor-grounded observability redesign.
```

The single allowed redesign is:

```text
score_pg(s,a) =
  normalized_predictor_uncertainty(s,a)
  * normalized_action_effect_observability(s,a)
```

where uncertainty is classifier entropy or margin on the train split, and
observability is the train-split action-effect detectability score. It must use
a shuffled-score control and the same `+0.10` threshold.

Stop rule:

```text
If predictor-grounded observability fails, stop ToyPowderWorld Stage C.
Do not run C0.7/C0.8 detector-tweak loops.
```

The next decision after failure is environment migration, target redesign, or a
project pause/rethink.

### Branch 3: Worst Case

Condition:

```text
oracle_event <= uniform
```

Decision:

```text
C0.5 detector failed.
Do not interpret observability.
Do not proceed to PSP.
```

Only one classifier-style detector redesign is allowed under this branch. If a
logistic/MLP detector with AUPRC/AUROC, class weighting, and alpha sweep cannot
make oracle_event win, ToyPowderWorld Stage C stops.

## Main Claim Rule

The full ICLR method framing requires both:

1. Stage B evidence remains stable: action-effect alignment exceeds
   static/shuffled controls.
2. Stage C evidence appears: observability or predictor-grounded weighting
   improves event/OOD prediction by at least `+0.10` and beats shuffled controls.

If condition 1 holds but condition 2 fails, do not force a method paper. The
remaining viable product is a diagnostic alignment paper or a redesigned
environment/model phase.

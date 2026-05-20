# Stage D0' Precommit: Predictor-Grounded Observability

Date: 2026-05-20

## Motivation

Stage B.6 found weak but stable action-effect alignment evidence. Stage C0 and C0.5 did not turn that signal into behavioral utility. In C0.5, even oracle-event weighting failed the oracle-positive detector gate. This means ToyPowderWorld Stage C should stop unless a new, explicitly preregistered phase is opened.

This document opens exactly one final Toy follow-up phase:

> Stage D0': predictor-grounded observability redefinition.

The goal is to test whether the problem is the current predictor-utility-naive
observability score, not to keep refining detectors indefinitely. This is an
explicit exception after the C0.5 stop decision. If D0' fails, do not create
C0.7/C0.8 or another Toy detector tweak.

## Hypothesis

The current score

```text
obs_geom = sqrt(rank(detect_rgb) * rank(detect_range))
```

measures action-effect detectability, but not whether a sample is useful for training the event/change predictor. Stage D0' tests a single predictor-grounded score:

```text
score_pg(s,a) = rank(entropy_uniform_predictor(s,a)) * rank(obs_geom(s,a))
```

Diagnostic variants may be reported, but the primary predictor-grounded score is entropy-times-observability.

## Methods

Primary comparison:

- `uniform`
- `change_mask`
- `observability`
- `shuffled_observability`
- `predictor_grounded`
- `shuffled_predictor_grounded`
- `oracle_event`

Primary target:

- `event_present`

Primary metrics:

- event AUPRC
- event AUROC
- event F1, selected by validation-threshold only
- balanced accuracy
- precision / recall

Primary behavioral effect size:

```text
method_delta_vs_uniform = method_metric - uniform_metric
```

The Stage-B6 reference signal is fixed at `0.05`.

## Branches

### Branch 1: Best case

Condition:

```text
oracle_event beats uniform by >= +0.10 AUPRC or F1
predictor_grounded beats uniform by >= +0.10 AUPRC or F1
predictor_grounded beats change_mask
predictor_grounded beats observability
predictor_grounded beats shuffled_observability
predictor_grounded beats shuffled_predictor_grounded
```

Decision:

```text
Proceed to Stage C1-lite with PSP-like comparison. Full PSP-like grids require
a separate preregistration, and D0' by itself does not establish OOD robustness.
```

### Branch 2: Median case

Condition:

```text
oracle_event beats uniform by >= +0.10,
but predictor_grounded does not beat uniform/change_mask/shuffled by >= +0.10.
```

Decision:

```text
Stop ToyPowderWorld Stage C. Do not run C0.7/C0.8 detector tweaks.
Open Stage D0 environment migration only if the project still wants to invest.
```

### Branch 3: Worst case

Condition:

```text
oracle_event does not beat uniform by >= +0.10.
```

Decision:

```text
ToyPowderWorld remains unsuitable for behavioral validation under the current target/model setup.
Stop Toy Stage C. Any further work requires a new preregistered target/model/environment phase.
```

## Effect-size thresholds

- `>= +0.15`: strong behavioral validation.
- `>= +0.10`: sufficient to justify Stage C1.
- `+0.05 to +0.10`: weak / inconclusive. Do not proceed to PSP.
- `< +0.05`: No-go.

## Main-claim rule

The full method-paper framing is only allowed if both are true:

1. Stage B evidence remains stable: action-effect alignment > static/shuffled.
2. Stage D0'/C evidence appears: predictor-grounded or observability weighting
   improves event prediction by at least +0.10 over uniform and beats
   change-mask and shuffled controls. OOD robustness is not tested in D0' and
   must be handled in a later preregistered Stage C1-lite/C1 experiment.

If Stage B holds but Stage D0' fails, do not force a method paper.

## Hard-gate interpretation

The oracle gate is a hard gate. If `oracle_event` fails to beat uniform by at
least `+0.10` on event AUPRC or event F1, all predictor-grounded and
observability results are diagnostic-only, even if numerically positive.

The primary predictor-grounded score is exactly:

```text
rank_train(entropy_uniform_predictor) * rank_train(obs_geom)
```

where `obs_geom` is the geometric mean of train-fitted detectability ranks.
Diagnostic variants such as predictor uncertainty or loss-gradient
observability are not eligible for Branch 1. `lossgrad_observability` uses
labels and is oracle-like diagnostic only.

Best-alpha selection is allowed only within this preregistered alpha sweep, but
the winning row must be stable: it must be positive in most split seeds and beat
the best non-oracle controls on the same behavioral metric. A single best-alpha
positive result is not sufficient.

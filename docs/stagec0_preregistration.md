# Stage C0 Preregistration: Minimal Selective-Prediction Smoke

## Status

Stage C0 is **not** a PSP/Dreamer comparison and not a full world-model paper. It is a behavioral smoke test for the Stage-B6 finding:

> Action-effect signatures show weak but reproducible cross-channel structure, and continuous action-effect observability is positively associated with alignment.

The purpose is to test whether this weak diagnostic signal has predictive utility.

## Main question

Does continuous action-effect observability, used as a sample-weighting signal, improve downstream prediction metrics beyond uniform reconstruction and simple change-mask weighting?

## Hypotheses

### H0: kNN-only signal

The Stage-B6 observability signal exists only in the kNN measurement primitive. Observability weighting will not beat uniform or change-mask training on event/OOD metrics.

### H1: behavioral utility

The Stage-B6 observability signal reflects useful action-effect geometry. Observability weighting will improve event prediction, changed-cell prediction, or OOD robustness. The effect size may be larger than the Stage-B6 kNN signal (~0.05).

## Methods

Stage C0 trains simple weighted ridge prediction heads on Stage-0 dense samples. It uses static pre-action observations and action features as input, and predicts future action-induced deltas/events.

Primary methods:

| Method | Purpose |
|---|---|
| `uniform` | baseline reconstruction objective |
| `change_mask` | observation-level dynamics-aware baseline |
| `observability` | proposed C0 signal from detectability ranks |
| `shuffled_observability` | sanity control with same marginal weights |
| `oracle_event` | optional upper bound / target-leakage diagnostic |

Primary targets:

- full action-induced reconstruction delta for `target_channel`;
- future event response from `event_response` as target only, never as input;
- changed-cell map from action-induced target-channel deltas;
- OOD distractor robustness by perturbing RGB inputs at test time.

## Primary input policy

Default input is `rgb` + action features. A secondary grid may include `rgb range`, but the first C0 claim should not depend on privileged or diagnostic response channels.

`event_response` is a target/diagnostic response and must not be used as input.

## Primary observability score

Default:

```text
observability = sqrt(rank_train(detect_rgb) * rank_train(detect_range))
```

Additional sweeps:

```text
geom / mean / product / min
```

The ranks are fit inside each train split. Validation/test rows are not used to
set the training-weight quantiles. This makes Stage C0 an oracle training-weight
smoke, not a deployable proxy.

## Required output columns

The summary table must include:

- full reconstruction MSE;
- event prediction F1;
- OOD event F1;
- changed-cell F1;
- absolute effect size relative to uniform;
- effect size minus the Stage-B6 reference signal (~0.05);
- effect size divided by the Stage-B6 reference signal.
- weight effective sample size and top-decile mass.

This is required because a +0.03 to +0.05 improvement is only an echo of the Stage-B kNN effect. A +0.10 or larger behavioral effect is a substantially stronger signal.

## Go condition for Stage C1

Proceed to PSP-like comparisons only if:

1. `observability` beats `uniform` on event F1 or OOD event F1;
2. `observability` beats or clearly differs from `change_mask`;
3. `shuffled_observability` does not match the proposed signal;
4. full reconstruction can still favor `uniform`, while event/OOD favors `observability`;
5. the effect-size delta on at least one behavioral metric is meaningfully larger than the Stage-B6 kNN reference signal (~0.05).

## No-go condition

Do not start PSP/Dreamer comparisons if:

- observability weighting does not beat uniform;
- observability and shuffled observability are indistinguishable;
- the only positive result is reconstruction MSE;
- the behavioral effect size remains around +0.03 to +0.05 with no OOD/event amplification.

## Interpretation

A positive C0 result supports the working thesis:

> Action-effect observability is not merely a kNN diagnostic; it can guide what a predictive model should emphasize.

A negative C0 result means the Stage-B signal is likely measurement-local and not yet behaviorally useful. In that case, do not force a paper. Reconsider the environment, representation primitive, or framing.

## Leakage Guards

- `event_response`, `semantic`, and `edge` are rejected as C0 input channels.
- `event_response` must exist as a target for primary runs; falling back to
  `world_delta` requires an explicit diagnostic flag.
- Event and changed-cell F1 are scored using predicted magnitude, not signed raw
  deltas, because the binary targets are defined from absolute changes.

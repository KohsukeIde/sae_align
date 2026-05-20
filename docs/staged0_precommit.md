# Stage D0 Precommit: real Powderworld oracle-positive detector audit

Stage D0 is a new preregistered phase after ToyPowderWorld Stage C0/C0.5/D0' No-go.
It is **not** PSP, Dreamer, RL training, or a method paper yet.

## Investment upper bound

D0 is a one-phase feasibility audit. If D0 does not pass both gates below, we do
not continue into D1/D2/C1 loops for the current theme.

This means:

- no PSP-like comparison unless both oracle and observability/predictor-grounded gates pass;
- no Dreamer-like comparison in D0;
- no repeated detector tweaks after a median or worst-case D0 outcome;
- if D0 fails, the current theme's ICLR-oral path is closed, not the overall research ambition.

## Generator validity gate

Before Branch 1/2/3 is interpreted, the real-Powderworld generator must pass a
sanity gate. This is not a method result.

Required:

- event-present prevalence must be non-degenerate: `0.02 <= prevalence <= 0.98`;
- `world_delta` must be positive for a non-trivial fraction of rows;
- RGB/range/local detectability summaries must be written and inspected;
- Toy-vs-real sanity summaries should show that real Powderworld is not simply
  reproducing the ToyPowderWorld action-effect distribution.

If this gate fails, the result is `generator_invalid`, not Branch 1/2/3. The
correct response is a separately preregistered environment/target reset, not
detector tuning.

## Branch 1: oracle and observability pass

Criteria:

- `oracle_event` or equivalent oracle task weighting beats uniform by at least `+0.10` on event AUPRC or event F1;
- `observability` or `predictor_grounded` also beats uniform by at least `+0.10`;
- the winner beats `change_mask` and shuffled controls.
- the oracle and winning method pass stably across the precommitted independent
  generator seeds, not only one favorable split.

Decision: proceed to Stage C1 / PSP-like comparison.

## Branch 2: oracle passes, observability fails

Criteria:

- oracle weighting beats uniform by at least `+0.10`;
- observability / predictor-grounded fails to reach `+0.10`, or is matched by shuffled controls.

Decision:

- stop Stage C for this environment;
- do not tune C0.7/C0.8/D1/D2 detectors;
- choose between diagnostic write-up, a separately preregistered new project, or
  major reframe. A new observability concept is not a continuation of this D0
  result and must use fresh preregistration/data.

## Branch 3: oracle fails

Criteria:

- oracle weighting fails to beat uniform by `+0.10`.

Decision:

- environment/target/model is not usable for current behavioral validation;
- do not interpret observability;
- do not proceed to PSP-like comparison.
- do not tune D1/D2/C0.7/C0.8 detector variants on this failure.

## Effect-size threshold

The minimum Stage-C1 threshold is `+0.10` absolute improvement on event AUPRC or event F1.
`+0.05` is not enough, because Stage B6 kNN alignment signals were roughly `+0.03` to `+0.05`.

## Main claim if D0 succeeds

The full theme can only proceed if the following two-part claim becomes supported:

1. Action-effect signatures reveal weak but stable cross-channel structure beyond static representations.
2. In a richer Powderworld environment, observability or predictor-grounded observability converts this structure into selective prediction utility beyond uniform/change-mask baselines.

If claim 2 fails, do not force a method paper.

# Project Pivot Memo: From Observability Weights to Action Instruments

This memo is for repo-internal context only.  It is not intended as SoP text, paper text, or public framing.

## What survived

Stage B.6 left weak but stable diagnostic evidence:

- `rgb-range` action-effect alignment remained positive across seeds/splits.
- The effect size was small, approximately `+0.03` to `+0.05`.
- Literature-style local graph metrics, especially CKNNA, supported action-effect alignment while static/shuffled graph controls were near null.

A conservative surviving claim is:

> Action-effect signatures expose weak but stable cross-channel structure beyond static representations.

## What failed

The scalar observability-weighting route failed.

- Stage C0: weighted ridge did not show useful behavioral validation; oracle event weighting did not beat uniform.
- Stage C0.5: logistic detector still did not produce an oracle-positive behavioral gate.
- Stage D0': predictor-grounded observability had a raw positive but was not interpretable because oracle failed.
- Stage D0 real Powderworld: generator sanity passed, but oracle-event again failed the hard gate.

Therefore the repo should not continue with C0.7/C0.8/D1/D2 detector tweaks under the same theme.

## Interpretation

One plausible method error was collapsing a structured action-effect signature into a scalar sample weight.  This is a working hypothesis, not something proved by D0.

B.6 measured a graph/subspace structure:

```text
D_m(s) = [Delta z_m(s, a_1), ..., Delta z_m(s, a_K)]
```

C0/D0 converted that into a row weight:

```text
w(s,a) = scalar observability score
```

This changed the object of study and may have hidden the weak B.6 signal.  It is also possible that the B.6 signal is simply too weak to support a method contribution.  The Action-IV phase is therefore a fresh feasibility test, not a rescue claim.

## New working hypothesis

> Actions are instruments, not weights.

Action-induced changes should be used to identify a shared effect subspace across observation channels.  Until formal instrumental-variable assumptions are tested, this should be described operationally as an action-conditioned paired-effect representation, not as a formal IV estimator.

## What not to claim

Do not claim:

- behavioral utility from observability weighting;
- binary regular/blind strata as the primary explanatory mechanism;
- PSP/Dreamer competitiveness;
- that current D0/D0' results support a method paper.
- that Action-IV is a formal instrumental-variable method without explicit relevance, exclusion, and independence tests.

## Reusable infrastructure

Keep:

- Stage 0 channel sanity protocol;
- action-effect signature tooling;
- local graph alignment metrics;
- Powderworld adapter skeleton;
- precommit/branch discipline.

Discard or demote:

- scalar observability weighting as the main method;
- ToyPowderWorld Stage C detector loop.

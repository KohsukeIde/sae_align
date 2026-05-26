# Framing D / AERA Precommit

Date: 2026-05-26 JST

## Scope

This document opens a narrow experimental path after the contribution-type
pause.  The survey itself is handled separately.  This precommit covers only
what can be run experimentally before any AERA prototype is implemented.

## Thesis

```text
The convergent object across observation channels is not the static
representation, but the action-conditioned effect representation.
```

Working name:

```text
AERA: Action-Effect Representation Alignment
```

## What AERA Is

AERA would learn effect representations

```text
e_m(s, a) = f_m(o_m(s), o_m(s^a), a)
```

and align `e_rgb(s,a)` with `e_range(s,a)` for the same state-action
intervention.  The central object is the paired action effect, not a scalar
sample weight.

## What AERA Is Not

- not scalar observability weighting;
- not PSP or Denoised MDP;
- not generic CURL / ATC / TACO-style single-view contrastive RL;
- not a full causal-variable-identification claim;
- not a custom simulator paper;
- not a revival of C0/C0.5/D0 detector tweaks.

## Custom Simulator Rule

No new custom simulator may be used as main evidence.  Custom environments are
allowed only as debugging or appendix sanity checks after the main existing
environment evidence is established.

Main experimental priority:

```text
real Powderworld B6-style action-effect alignment
```

## Phase 1: Real Powderworld B6-Style Alignment

Purpose:

```text
Test whether the ToyPowderWorld B6 signal survives in a public existing
environment using the same counterfactual do-action vs no-op protocol.
```

Inputs:

- real Powderworld dynamics;
- RGB and range-like channels;
- do-action and no-op rollouts;
- fixed action bank shared across states;
- probe/held-out action split;
- redundancy controls when available (`noisy_rgb`, `gray_rgb`, `blur_rgb`).

Primary comparisons:

- static alignment;
- action-effect alignment;
- shuffled-action / shuffled-effect controls;
- redundancy controls;
- CKNNA / cycle-kNN as diagnostic local-graph metrics;
- continuous observability correlation as secondary diagnostic.

Primary target pair:

```text
rgb:range
```

### Phase-1 Go

Proceed toward AERA method specification only if real Powderworld shows:

```text
action-effect > static
action-effect > shuffled
rgb-range adjusted effect size >= +0.05
CKNNA positive with static/shuffled near null
redundancy controls pass if included
```

### Phase-1 Partial

If:

```text
+0.03 <= rgb-range adjusted effect size < +0.05
```

then the result is partial.  Do not implement AERA immediately.  First decide
whether task suitability audit is needed to find an evaluation environment.

### Phase-1 No-Go

If:

```text
rgb-range adjusted effect size < +0.03
or action-effect ~= static
or action-effect ~= shuffled
```

then:

```text
Do not build AERA prototype.
Do not build custom simulator.
Reframe as diagnostic/analysis paper or terminate this route.
```

## Phase 2: AERA Novelty Survey

The novelty survey is separate from this repo work, but AERA implementation is
blocked until it is complete.

The survey must establish technical distinction from:

- weakly supervised causal representation learning;
- multi-view causal representation learning;
- CURL / ATC / TACO and temporal/action contrastive RL;
- PSP / Denoised MDP / selective prediction methods.

Go:

```text
technical difference is clear enough to write an AERA method section
```

No-go:

```text
AERA is essentially an existing CRL / multi-view / contrastive objective
```

## Phase 3: AERA Prototype

Blocked until both Phase 1 and Phase 2 pass.

Prototype Go thresholds, if opened later:

```text
cross-modal effect retrieval >= static baseline + 0.15
held-out action retrieval >= shuffled + 0.10
task/effect head improves over action-only or raw-delta baseline
```

No-go:

```text
Do not run AERA-v2 / loss-tweak loop.
```

## Role of Action-IV Step 2 Failure

Under Framing D, the Action-IV DestroyAll Step-2 failure is not central evidence
against AERA.

Its role is:

```text
Scalar sample weighting is the wrong bridge from action-effect diagnostics to
behavioral utility.
```

The postmortem already established:

- DestroyAll is learnable by the current obs+action detector;
- oracle-as-feature reaches AUPRC `1.0`;
- action-only is strong;
- oracle sample/class weighting does not improve AUPRC ranking.

This motivates representation-level effect alignment.  It does not authorize
Action-IV Step 3, and it does not revive scalar weighting.

## Path-Building / Sand-Pushing Audit Rule

Whether to run task suitability audit depends on the real Powderworld B6 result.

### Case 1: Clear B6 Pass

Condition:

```text
rgb-range adjusted effect size >= +0.05
CKNNA replicates
controls pass
```

Action:

```text
Do not run Path-Building / Sand-Pushing audit now.
Proceed with Framing D primary after novelty survey.
Use Action-IV Step-2 failure only as scalar-weighting motivation.
```

### Case 2: Partial B6 Pass

Condition:

```text
+0.03 <= rgb-range adjusted effect size < +0.05
```

Action:

```text
Run task suitability audit only to select an AERA evaluation environment.
The audit is not central evidence for scalar-weighting failure.
```

If Path-Building or Sand-Pushing is obs-critical and less action-shortcut
dominated, AERA may later be evaluated there after a new prototype precommit.

### Case 3: B6 Fail

Condition:

```text
rgb-range adjusted effect size < +0.03
```

Action:

```text
Stop Framing D / AERA.
Do not implement AERA.
If continuing the project at all, switch to an analysis/evaluation fallback.
```

Under that fallback, Path-Building / Sand-Pushing audit may become central
evidence only if a new analysis-paper precommit says so.

## Forbidden Until Phase-1/Phase-2 Pass

- AERA implementation;
- Action-IV Step 3;
- custom simulator development;
- PSP / Dreamer / RL comparisons;
- Path-Building / Sand-Pushing generation, except under the Case-2/Case-3 rules
  above;
- loss-function tweaks under the AERA name.


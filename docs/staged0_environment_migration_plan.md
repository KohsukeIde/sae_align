# Stage D0 Plan: Environment Migration / Oracle-Positive Detector Audit

This document is a plan, not an implementation.

## Why not jump directly to full PSP/Dreamer?

Stage C0/C0.5 failed on ToyPowderWorld. Before any PSP/Dreamer-style comparison, a new environment must first pass an oracle-positive detector audit.

## Candidate environment

Preferred first candidate: original Powderworld.

Powderworld is designed as a lightweight 2D environment with modular local interactions and rich task distributions. It is therefore a plausible next step after the ToyPowderWorld diagnostic environment.

## Repo-structure principle

Do not replace the current toy pipeline. Add an adapter layer:

```text
src/sae_align/envs/powderworld_adapter.py
scripts/make_staged0_dataset.py
scripts/train_staged0_oracle_audit.py
configs/staged0_powderworld.json
```

This mirrors common MBRL repository organization: keep experience collection / data generation, model fitting, evaluation, and logging separated.

## D0 gate

First run only oracle-positive detector audit:

- `uniform`
- `oracle_task` / `oracle_event`
- `change_mask`
- `observability`
- `shuffled_observability`

Go condition:

```text
oracle_task or oracle_event beats uniform by >= +0.10 AUPRC/F1.
```

If oracle fails, the environment/target is not suitable for behavioral validation. Do not evaluate observability.

## D0 branch 2

If oracle wins but observability loses, do not keep tweaking detector heads. Either:

1. use the predictor-grounded D0' score exactly once in the migrated environment; or
2. stop the method-paper route and reconsider the project scope.

## Relation to D0'

D0' is cheaper and should be run before or in parallel with D0. If D0' fails,
D0 must be justified as an environment-richness bet, not as a continuation of
detector tweaking.

The first migrated-environment experiment must preserve the D0 audit order:

1. establish an oracle-positive detector;
2. only then evaluate observability or predictor-grounded observability;
3. do not tune target, alpha, head, threshold policy, or score definition using
   the D0' failure pattern.

If the migrated environment also fails the oracle-positive audit, observability
is not interpretable there either.

## Post D0' v1 Decision

D0' v1 failed the oracle-positive gate. Predictor-grounded observability showed
a raw positive AUPRC direction, but it is not interpretable because
`oracle_event` did not beat `uniform` by the precommitted threshold.

Decision:

- ToyPowderWorld Stage C stops.
- Do not add C0.7/C0.8 or another Toy detector tweak.
- D0 may proceed only as a separately preregistered environment-richness audit,
  not as a continuation of D0' score tuning.

The first D0 audit must freeze the environment config, target/event definition,
detector head, alpha grid, split policy, metrics, and threshold policy before
looking at migrated-environment results.

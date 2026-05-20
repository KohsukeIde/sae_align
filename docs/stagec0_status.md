# Stage C0 Status and Next Actions

## Current state before C0

- Stage 0: passed channel sanity.
- Stage B.1-B.4: diagnostic/metric repair.
- Stage B.5: partial positive.
- Stage B.6: partial robust positive.
- Binary regular/blind framing: weak.
- Continuous action-effect observability framing: current working hypothesis.
- Full PSP/Dreamer comparisons: still premature.

## Stage C0 v1 Result

The first hardened C0 v1 run is recorded in
`docs/stagec0_v1_cpu_experiment.md`.

Summary:

- `3` data seeds, `8` input/mix cells per seed, `24` total cells;
- all final qsub tasks exited with status `0`;
- observability weighting did not beat uniform on event F1 or OOD event F1 in
  any cell;
- the primary-like `input_rgb / obs_geom` cell had event F1 delta vs uniform
  `-0.0408` and OOD event F1 delta vs uniform `-0.0449`;
- `oracle_event` weighting also failed to beat uniform, suggesting the current
  weighted-ridge C0 setup may be a weak detector of useful sample weighting.

Decision:

```text
Stage C0 v1: No-go.
Stage C1 PSP-like comparison: blocked.
Next action: redesign C0 target/model/weighting before any PSP/Dreamer work.
```

## Stage C0.5 v1 Result

The oracle-positive detector redesign is recorded in
`docs/stagec05_precommit.md` and `docs/stagec05_v1_cpu_experiment.md`.

Summary:

- C0.5 added a logistic classifier detector with AUPRC/AUROC/balanced accuracy,
  alpha sweeps, tie-aware ranks, and explicit binary oracle weights.
- Event-present detector: observability improved AUPRC weakly (`+0.0723` best)
  but did not improve F1, did not reach the `+0.10` threshold, and was similar
  to change-mask.
- Oracle-event weighting failed the oracle-positive gate; event AUPRC deltas
  were negative for oracle-event weighting.
- Changed-any target was saturated; oracle/change/observability weights changed
  AUPRC/F1 by about `0.001` or less.

Decision:

```text
Stage C0.5 v1: No-go.
Stage C1 PSP-like comparison: still blocked.
C0.6 predictor-grounded redesign: not triggered under the precommit, because
the oracle-event detector did not pass.
ToyPowderWorld Stage C should stop unless a new preregistered target/model or
environment migration phase is chosen.
```

Stage D0' was the one explicit exception opened after this stop decision. It is
recorded in `docs/staged0p_v1_cpu_experiment.md`.

Stage D0' v1 was also No-go: `oracle_event` failed the hard oracle-positive
gate, while `predictor_grounded` produced only a diagnostic raw AUPRC lift.
ToyPowderWorld Stage C is now stopped. Do not add C0.7/C0.8 detector-tweak
loops.

## What C0 changes

Previous stages were inward-facing diagnostics: kNN alignment, action split reliability, PCA/tie sensitivity, and observability correlation. Stage C0 is outward-facing: it asks whether observability weighting improves prediction behavior.

## Commands

Smoke:

```bash
PYTHONPATH=src bash scripts/run_stagec0_smoke.sh outputs/stagec0_smoke
```

V1 CPU grid:

```bash
PYTHONPATH=src bash scripts/run_stagec0_v1_cpu.sh outputs/stagec0_v1_cpu outputs/stage0_v1/data/stage0_dataset.npz
```

Aggregate:

```bash
PYTHONPATH=src python scripts/summarize_stagec0_grid.py --root outputs/stagec0_v1_cpu
```

## Read first

- `docs/stagec0_preregistration.md`
- `reports/stagec0_summary.csv`
- `reports/stagec0_decision_summary.json`
- `stagec0_method_delta_summary.csv` after grid aggregation

## Important caveats

- Stage C0 uses simple ridge heads. A negative result does not prove the final idea is false, but it blocks PSP/Dreamer comparisons.
- `event_response` is a target/diagnostic response, not an input channel.
- `oracle_event` is an upper-bound/leakage diagnostic, not a deployable method.
- Stage C0 should not be used to claim ICLR-level method performance by itself.
- The primary observability score is a train-split geometric detectability score,
  matching the B.6 continuous-observability direction more closely than a global
  row-rank mean.
- The bundled qsub script is ABCI/PBS-compatible and is intended for grids that
  exceed a short local smoke run.

## Decision rule

This section is now superseded by the completed C0.5, D0', and D0 audits.

Current final rule:

- Stage C1 / PSP-like comparison is blocked.
- ToyPowderWorld Stage C is stopped.
- Stage D0 real-Powderworld audit was also No-go.
- Do not continue with C0.7/C0.8/D1/D2 detector-tweak loops for the same
  theme.
- Any further work must be a separately preregistered target/model or project
  redesign, or a reframing around diagnostic alignment evidence rather than
  behavioral utility.

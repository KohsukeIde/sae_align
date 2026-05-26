# Changelog

## Unreleased

- Repositioned `edge` as a derived Stage 0 diagnostic channel rather than a primary redundancy control.
- Documented `noisy_rgb`, `gray_rgb`, and `blur_rgb` as the primary RGB redundancy controls.
- Clarified that `event_response` is diagnostic-only and excluded from primary Stage B unless explicitly run as an `--allow-leakage-diagnostic` leakage diagnostic.
- Added Stage 0 pass-to-Stage-B criteria and next-step framing.
- Documented the current NumPy Stage B pilot as regular/blind pairwise alignment with required action-only and shuffled controls.
- Clarified that current Stage 0/B pilots do not require qsub, a GPU, or CUDA.
- Added Stage B.2 state-level action-effect signature smoke/pilot support with full-state dense sampling, probe-action encoder training, held-out action scoring, fingerprint checks, and static baselines.
- Added Stage B.2 stratum-fraction summaries and optional query-bootstrap confidence interval reports.
- Recorded the Stage B.2 v1 local CPU experiment and its decision: weak partial signal, not pass; Stage C remains blocked.
- Added Stage B.3 preregistration, balanced action-split generation, and gate-analysis scripts for normalization, redundancy transfer, static/shuffled controls, and continuous observability diagnostics.
- Recorded the Stage B.3 v1 local CPU grid: diagnostic-only, redundancy probe-to-heldout calibration failed, Stage C remains blocked.
- Added Stage B.4 split-half reliability gates, identity checks, same-channel
  versus shuffled paired bootstrap CI, feature tie diagnostics, and grid
  aggregation.
- Updated balanced action split scoring to use primary-channel detectability
  and conditional blindness while excluding diagnostic/oracle channels from
  split optimization.
- Recorded Stage B.4 v1 local CPU grid: identity passed, but same-channel
  reliability and redundancy calibration failed under CI; Stage C remains
  blocked.
- Added Stage B.5 held-out same-action-set preregistration, analyzer, smoke/v1
  scripts, and grid summarizer.
- Recorded Stage B.5 v1 local CPU grid: held-out redundancy controls passed and
  `pca_probe_only / probe_action_type_apply` showed a partial positive
  `rgb-range` signal, but the result is representation/normalization dependent;
  Stage C remains blocked.
- Added Stage B.6 preregistration, artifact diagnostics, calibrated
  CKA/RSA/ridge measurement sanity checks, k/jitter/PCA-dimension grid runners,
  and ABCI/qsub CPU-array helpers.
- Added `docs/alignment_metric_notes.md`, recording the literature-method
  review behind Stage B.6: mutual kNN and gallery-density critiques,
  permutation-calibrated CKA/RSA/neighborhood metrics, and CCA/subspace
  retrieval diagnostics.
- Recorded Stage B.6 v1 CPU array: the preregistered primary cell reproduced a
  weak positive `rgb-range` signal and continuous observability association;
  label is partial robust positive, not full pass.
- Added diagnostic-only Stage B.6 literature metrics: cycle-kNN, CKNNA,
  CCA, and SVCCA, written separately from `b6_measurement_sanity.csv`.
- Added B.6 primary-cell larger CPU-array scripts for `3 x 3` seed/split
  replication over PCA dims `16` and `32` with 256 complete states.
- Recorded the B.6 primary-cell larger CPU experiment: 18/18 tasks completed,
  fixed `d=32/k=10/jitter=0` primary cell stayed positive in 9/9 runs, and
  CKNNA provided the cleanest diagnostic support among literature metrics.
- Added and ran Stage C0 v1. The implementation was hardened against
  train/test ranking leakage, diagnostic-channel input leakage, silent event
  fallback, and signed-delta F1 scoring. The first 3-data-seed C0 grid was
  No-go: observability weighting did not beat uniform on event/OOD F1, and
  oracle-event weighting also failed to beat uniform.
- Added Stage C0.5 precommit and logistic detector smoke. C0.5 added
  tie-aware ranks, explicit binary oracle weights, AUPRC/AUROC/balanced
  accuracy, alpha sweeps, and changed-any diagnostics. The v1 detector was
  No-go: oracle-event weighting failed the oracle-positive gate, observability
  produced only a weak AUPRC lift below `+0.10`, and changed-any was saturated.
- Added Stage D0' as the final one-shot ToyPowderWorld follow-up before
  environment migration or project redesign. D0' tests exactly one
  predictor-grounded observability score, treats oracle-event as a hard gate,
  and blocks C0.7/C0.8 detector-tweak loops.
- Ran Stage D0' v1 locally on the three default B.6 primary-cell datasets. The
  result was No-go: predictor-grounded observability produced a raw AUPRC lift
  (`+0.0898`) but oracle-event failed the hard gate (`+0.0015` behavior delta),
  so ToyPowderWorld Stage C is stopped.
- Added Stage D0 real-Powderworld oracle-positive detector audit files:
  optional real adapter, D0 dataset generator, toy smoke, sanity comparison,
  qsub CPU-array submission, and D0 precommit/status docs. D0 remains an
  oracle-positive feasibility audit only; it is not PSP/Dreamer/RL.
- Hardened the D0 generator and summary logic after review: event targets now
  exclude broad `any_change`, Toy backend subtracts no-op physics events,
  action-array width is inferred dynamically, do/no-op stochastic rollouts use
  paired seeds, generator sanity stats are emitted, event prevalence is gated,
  and D0 summaries use D0-specific branch labels with observability-or-PG pass
  logic.
- Ran Stage D0 v1 real-Powderworld CPU array over three generator seeds. Generator sanity passed, but the oracle-positive detector audit failed: `oracle_event` best AUPRC delta was `-0.0032`, best F1 delta was `+0.0068`, and the decision was Branch 3 / No-go. Stage C1 / PSP-like comparison remains blocked.
- Added the Action-IV pivot phase: repo-internal pivot memo, Action-IV
  precommit, official-task oracle sanity scripts, effect-subspace prototype,
  qsub helper, and Action-IV metrics.
- Hardened Action-IV after review: the real Powderworld backend now targets the
  installed `powderworld.envs`/`PWSim` API, action banks are fixed across
  states, DestroyAll audits include a predeclared erase-action fraction,
  degenerate targets are marked invalid, oracle alpha selection uses validation
  AUPRC only, and Step 3 is skipped unless Step 2 passes.
- Ran Action-IV Step-2 real-Powderworld DestroyAll v1b over three seeds. The
  target was non-degenerate, but the oracle-positive gate failed:
  validation-selected oracle AUPRC delta mean was `+0.00168` with min
  `-0.00052`. Action-IV Step 3 / neural prototype remains blocked for this
  task/model.
- Added and ran the Action-IV D0 gate postmortem on the existing v1b datasets.
  The audit found `case_a_task_learnable_weighting_gate_inappropriate`:
  oracle-as-feature sanity reached AUPRC `1.0`, uniform obs+action AUPRC was
  `0.9064` versus prevalence `0.2281`, but oracle sample weighting remained
  near-null. The old Step-2 gate remains failed, but the failure should be read
  as a bad oracle-weighting gate rather than an unlearnable DestroyAll target.
  Action-only AUPRC was also high (`0.8472`), so any Step-2-v2 / Step-3 task
  claim must include shortcut controls.
- Added `docs/experiment_pause_for_survey.md` to freeze experiment activity
  during the contribution-type survey. Immediate new experiments are blocked;
  Step-2-v2, Action-IV Step 3, new Powderworld tasks, and PSP/Dreamer
  comparisons require a survey-driven path decision and a fresh precommit.

## v0.1.0

- Initial starter repo.
- Toy Powderworld-like simulator.
- Stage 0 dataset generation.
- Channel adequacy analysis.
- Docs for channel specification and experiment roadmap.

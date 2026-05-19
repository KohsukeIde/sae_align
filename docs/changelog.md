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

## v0.1.0

- Initial starter repo.
- Toy Powderworld-like simulator.
- Stage 0 dataset generation.
- Channel adequacy analysis.
- Docs for channel specification and experiment roadmap.

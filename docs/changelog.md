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

## v0.1.0

- Initial starter repo.
- Toy Powderworld-like simulator.
- Stage 0 dataset generation.
- Channel adequacy analysis.
- Docs for channel specification and experiment roadmap.

# Stage B.4 Preregistration

Stage B.4 is a measurement-reliability gate, not a new paper framing.

## Motivation

Stage B.3 did not pass because redundancy controls failed probe-to-heldout
cross-action calibration under bootstrap CI. Therefore `rgb-range` cannot yet
be interpreted as evidence for or against the strata hypothesis.

## Gate Order

1. **Gate -1a: identity same-action test**
   Compare each channel against itself using the same action ID set through two
   independent feature-construction paths for both probe and held-out actions.
   Failure indicates a pipeline bug.
2. **Gate -1b: same-channel split-half reliability**
   Compare `D_m_probe(s)` and `D_m_heldout(s)` for each channel. If this fails,
   cross-channel probe-to-heldout alignment is not interpretable.
3. **Gate 0: redundancy cross-channel calibration**
   Only after Gate -1 passes, test `rgb-noisy_rgb`, `rgb-gray_rgb`, and
   `rgb-blur_rgb` probe-to-heldout calibration. `rgb-noisy_rgb` and
   `rgb-gray_rgb` are core redundancy controls; `rgb-blur_rgb` is reported as a
   broader redundancy / low-pass diagnostic.
4. **Gate 1: `rgb-range` action-effect > static**
   Use Stage B.3 metrics only after Gates -1 and 0 pass.
5. **Gate 2: binary vs continuous observability**
   Decide whether binary strata, continuous observability, or action-coupling
   framing is supported.

## Primary Channels

`rgb`, `range`, `local`, `noisy_rgb`, `gray_rgb`, and `blur_rgb`.
`event_response` remains diagnostic-only and is excluded.

## Split Balance

Action split selection balances action metadata plus channel-specific
detectability summaries:

- `mean_detect_*`
- `blind_given_physical_*`

These terms are used only to make probe and held-out action sets comparable.
They are not tuned on Stage B.4 alignment outputs.
Diagnostic/oracle channels are not used for split optimization.

## Pass / Fail Logic

- If identity fails, fix the pipeline.
- If same-channel split-half reliability fails, redesign action bank,
  signature, encoder fitting, or normalization.
- If same-channel passes but redundancy cross-channel fails, fix channel
  calibration before interpreting `rgb-range`.
- If redundancy passes, resume Stage B.3 Gate 1/2 under the passing
  normalizations.

Real Powderworld migration is deferred until this ToyPowderWorld metric
reliability gate is resolved. Moving environments now would confound metric
bugs with simulator limitations.

Identity pass is necessary but not sufficient. Same-channel split-half
reliability is the ceiling for any cross-channel claim.

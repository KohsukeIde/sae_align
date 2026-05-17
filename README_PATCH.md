# Stage 0 Patch Notes

This note records the documentation-level interpretation change for the Stage 0 patch package.

## What Changed

1. `edge` is repositioned as a derived diagnostic channel rather than the primary redundancy control.
2. `noisy_rgb`, `gray_rgb`, and `blur_rgb` are the primary RGB-derived redundancy controls.
3. The diagnostic event channel is named `event_response`; it is a diagnostic-only post-action response channel, not a primary Stage B channel or future model input.
4. Stage 0 is framed as a pass-to-Stage-B gate: pass only if the main channels show distinct blind loci and the RGB redundancy controls behave as expected under K/threshold/action-type checks.

## Interpretation

Low RGB-edge blind-locus overlap is not necessarily a failure. Edge is deterministic over the simulator state, but it isolates boundary changes rather than color changes. Use it to diagnose derived-view behavior, not to validate redundancy.

Primary redundancy checks should use:

- `noisy_rgb`
- `gray_rgb`
- `blur_rgb`

## Stage B Boundary

`event_response` should not be fed into a future predictive world model as if it were an ordinary observation available before prediction. If an event-based future input is needed, redesign it as a pre-action history channel such as `event_history`.

It is also excluded from primary Stage B reports. Include it only in separately labeled leakage diagnostics with the explicit `--allow-leakage-diagnostic` guard.

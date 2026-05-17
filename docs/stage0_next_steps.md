# Stage 0 Next Steps

## Positioning Update

Stage 0 is a channel-adequacy and stratification gate before later representation or world-model experiments. Its output should be read as a pass-to-Stage-B signal only when the observation channels and controls behave as intended under sweeps.

The main interpretation changes are:

1. `edge` is a derived diagnostic channel, not the primary redundancy control. It can isolate boundary changes and may have a blind locus distinct from RGB.
2. `noisy_rgb`, `gray_rgb`, and `blur_rgb` are the primary RGB redundancy controls.
3. `event_response` is a diagnostic-only post-action response channel. It is excluded from primary Stage B unless the run is explicitly labeled as a leakage diagnostic and uses the `--allow-leakage-diagnostic` guard.

## Stage 0 Pass-to-Stage-B Criteria

Proceed to Stage B design if:

1. `noisy_rgb`, `gray_rgb`, and `blur_rgb` behave as low-complementarity / high-redundancy controls with RGB.
2. The default trainable sensor-like channels have distinct blind-locus profiles: `rgb`, `range`, and `local`.
3. K-sweeps or K-bootstrap checks show that blind-locus assignments are not extremely action-subset dependent.
4. Threshold-specific matrices preserve the qualitative complementarity hierarchy.
5. Action-type reports do not show that one action family completely dominates the blind-locus structure.
6. `event_response` is reported as diagnostic-only and kept out of default Stage B trainable channels because it is a post-action response.

Do not require `edge` to have high blind overlap with RGB. Low RGB-edge overlap should be reported as a derived-view diagnostic result, not as a redundancy-control failure.

## Stage B Preparation

Stage B should use Stage 0 strata without redefining them from learned representations. The current primary Stage B report is regular/blind pairwise alignment over default trainable channels such as `rgb`, `range`, and `local`.

Stage B work should compare:

- action-effect features from no-op vs do-action deltas;
- query sets split by regular, modality-blind, and physical-null strata;
- action-only controls;
- shuffled-strata and shuffled-action controls.

Keep `event_response` out of primary Stage B reports and future world-model inputs unless the channel is redesigned so that it contains only information available before the predicted transition. Current Stage 0/B pilots are NumPy-only and do not require qsub, a GPU, or CUDA.

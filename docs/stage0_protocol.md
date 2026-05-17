# Stage 0 Protocol: Observation Channel Adequacy

Stage 0 verifies whether the proposed observation channels have non-redundant action-effect blind loci.  Do not start representation-alignment or world-model training experiments until this stage passes.

## Data unit

Each sample is a state-action pair `(s, a)`.  For each sample, we run two counterfactual rollouts:

- `do-action`: apply action `a`, then roll out for horizon `H`;
- `no-op`: apply no action, then roll out for the same horizon.

The action-induced world change is:

```math
\Delta x^\star(s,a) = x^{a}_{t+H} - x^{noop}_{t+H}.
```

For each channel `m`, the observation-level change is:

```math
\Delta o_m(s,a) = o_m(x^{a}_{t+H}) - o_m(x^{noop}_{t+H}).
```

This counterfactual definition is important. It prevents natural dynamics from being confused with action-induced change.

## Main strata

Main strata are representation-independent.

| Stratum | Definition | Interpretation |
|---|---|---|
| physical null | `E_x < eps_x` | the action did not meaningfully change the world state |
| modality blind | `E_x >= eps_x` and `d_m < tau_m` | the world changed, but channel `m` did not register it |
| regular | `E_x >= eps_x` and `d_m >= tau_m` | the world changed and channel `m` registered it |

`E_x` is physical effect magnitude. `d_m` is channel-level detectability. Thresholds are chosen by quantile sweeps.

## Required plots

1. Renderer examples.
2. Effect magnitude histograms for `E_x` and all `d_m`.
3. Blind-locus Jaccard matrix.
4. Complementarity matrix.
5. Redundancy-probe matrix.
6. K-sweep / threshold-sweep stability plots.

## Go conditions

Proceed if:

- trainable sensor-like channels have distinct blind loci;
- RGB plus `noisy_rgb`, `gray_rgb`, and `blur_rgb` behaves as low-complementarity / high-redundancy controls;
- `edge` is reported as a derived diagnostic channel, not as a primary redundancy control;
- `event_response` is reported as diagnostic-only and excluded from primary Stage B unless explicitly run as an `--allow-leakage-diagnostic` leakage diagnostic;
- semantic grid behaves like a privileged diagnostic channel;
- the hierarchy is stable under K-sweep and threshold-sweep;
- action-type reports do not show a single action family dominating the result;
- non-redundant channels are not easily predicted from RGB in the redundancy probe.

## No-go conditions

Stop if:

- all main blind loci are nearly identical;
- RGB-derived redundancy controls fail to behave redundantly;
- `event_response` is accidentally treated as a default Stage B trainable channel or future model input;
- primary Stage B conclusions rely on `event_response` leakage-diagnostic runs;
- conclusions disappear under small threshold changes;
- range/local responses are trivially predictable from RGB.

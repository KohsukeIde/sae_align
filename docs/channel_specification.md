# Channel Specification Memo

This document is intended to be fixed **before** running alignment or world-model training experiments.  If a channel definition is changed after seeing Stage 0 results, rerun Stage 0 and record the change in the changelog.

## Goal

The goal is to evaluate a spectrum of observation channels over the same latent dynamics, not to claim that every channel is a real-world sensor modality. The key quantity is whether a channel has non-redundant action-effect observability and a distinct blind locus. We use the following terms:

| Term | Meaning |
|---|---|
| observation channel | Any observation kernel over the same latent world state. |
| sensor-like channel | A channel with distinct action-effect observability and failure modes. |
| derived view | A channel deterministically derived from the same state or another view, e.g. boundaries or edges. |
| primary redundancy control | A channel intended to be redundant with RGB and to produce low complementarity, e.g. noisy/gray/blur RGB. |
| derived diagnostic channel | A deterministic derived view that may still isolate a different aspect of change, e.g. edge/boundary response. |
| privileged channel | A channel close to the simulator state, e.g. semantic grid. |
| representation family | Different encoders trained on the same observation channel. |

## Main channels

### 1. RGB

Global visual appearance rendering.  It is expected to be blind to low-contrast changes and to be vulnerable to visual corruption.

### 2. Range-like channel

Four directional range maps to blockers.  It is intentionally called **range-like**, not depth, because the toy environment is two-dimensional. Glass is treated as transparent to produce a failure mode distinct from RGB.

### 3. Local interaction channel

Action-site-centered patch containing local material response channels: hardness, temperature, fluidity, and solidity. This is agent/action-centric and expected to be blind to distant events.

### 4. Event-response channel

Spatially marginalized post-action endogenous event-count vector over a rollout. This is **not** audio. It contains endogenous interaction-event types and rates, but no spatial location. Direct intervention events such as place/erase/push are excluded to avoid direct action leakage. It is expected to be blind to spatial localization.

This channel is named `event_response` to clarify that it is a diagnostic-only post-action response channel. It is excluded from primary Stage B reports and future world-model inputs. Include it only in a separately labeled leakage diagnostic, and require the explicit `--allow-leakage-diagnostic` guard for Stage B pilot runs.

## Control channels

### Semantic grid

Privileged diagnostic/upper-bound channel. It is not a main modality.

### Edge map

Boundary-derived diagnostic channel. It is deterministic over the simulator state and is derived-view-like, but it is not the primary redundancy control. It isolates boundary change rather than color change, so it can have blind loci distinct from RGB. This distinction is useful: deterministic derivation from the same latent state does not necessarily imply shared blind locus.

### Noisy RGB copy

Primary redundancy control. It should behave similarly to RGB and should not produce strong fusion gain merely by increasing channel count.

### Gray RGB

Primary redundancy control. Linear grayscale projection of RGB.

### Blur RGB

Primary redundancy control. Local blur of RGB.

## Initial action bank

The initial bank contains local placement, deletion, and push-like interventions. Analyses must include K-sweeps over action-bank subsets to check that blind-locus estimates are not action-sampling artifacts.

## Fixed design assumptions

- Main strata are defined using oracle world state and observation-level detectability only.
- Representation-blind regions are diagnostic only and not part of the main Stage 0 strata.
- All channel thresholds are selected via within-channel quantiles, not raw cross-channel norms.
- Complementarity must be evaluated with held-out actions before being connected to fusion gain.
- Primary redundancy controls are `noisy_rgb`, `gray_rgb`, and `blur_rgb`. Edge is a derived diagnostic channel.
- Primary Stage B alignment reports use regular/blind pairwise alignment over trainable channels such as `rgb`, `range`, and `local`; `event_response` is diagnostic-only.

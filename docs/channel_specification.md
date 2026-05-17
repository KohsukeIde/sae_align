# Channel Specification Memo

This document is intended to be fixed **before** running alignment or world-model training experiments.  If a channel definition is changed after seeing Stage 0 results, rerun Stage 0 and record the change in the changelog.

## Goal

The goal is to evaluate a spectrum of observation channels over the same latent dynamics, not to claim that every channel is a real-world sensor modality.  We use the following terms:

| Term | Meaning |
|---|---|
| observation channel | Any observation kernel over the same latent world state. |
| sensor-like channel | A channel with distinct action-effect observability and failure modes. |
| derived view | A channel nearly deterministically derived from another channel, e.g. edge from RGB. |
| privileged channel | A channel close to the simulator state, e.g. semantic grid. |
| representation family | Different encoders trained on the same observation channel. |

## Main channels

### 1. RGB

Global visual appearance rendering.  It is expected to be blind to low-contrast changes and to be vulnerable to visual corruption.

### 2. Range-like channel

Four directional range maps to blockers.  It is intentionally called **range-like**, not depth, because the toy environment is two-dimensional. Glass is treated as transparent to produce a failure mode distinct from RGB.

### 3. Local interaction channel

Action-site-centered patch containing local material response channels: hardness, temperature, fluidity, and solidity. This is agent/action-centric and expected to be blind to distant events.

### 4. Global event channel

Spatially marginalized event-count vector over a rollout. This is **not** audio. It contains endogenous interaction-event types and rates, but no spatial location. Direct intervention events such as place/erase/push are excluded to avoid action leakage. It is expected to be blind to spatial localization.

## Control channels

### Semantic grid

Privileged diagnostic/upper-bound channel. It is not a main modality.

### Edge map

RGB/semantic-derived negative control. It should be redundant with RGB and should not produce strong complementarity.

### Noisy RGB copy

Redundancy control. It should behave similarly to RGB and should not produce strong fusion gain merely by increasing channel count.

## Initial action bank

The initial bank contains local placement, deletion, and push-like interventions. Analyses must include K-sweeps over action-bank subsets to check that blind-locus estimates are not action-sampling artifacts.

## Fixed design assumptions

- Main strata are defined using oracle world state and observation-level detectability only.
- Representation-blind regions are diagnostic only and not part of the main Stage 0 strata.
- All channel thresholds are selected via within-channel quantiles, not raw cross-channel norms.
- Complementarity must be evaluated with held-out actions before being connected to fusion gain.


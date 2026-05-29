# Static-Weak Real Powderworld v2 Experiment

Date: 2026-05-29 JST

## Purpose

This exploratory run tests whether the real-Powderworld `rgb:range` result is
static-dominated because that channel pair shares scene layout/occupancy.  It
does not authorize AERA implementation.  It can only nominate a frozen cell for
later preregistered replication.

The `v1` attempt failed before scientific interpretation because the original
real-Powderworld action bank sampled uniformly over material elements, making
erase actions too rare for `probe_action_type_apply`.  The `v2` dataset uses an
explicitly mixed erase/place action bank (`erase` probability 0.25).

## Setup

```text
output root: outputs/static_weak_realpw_v2_cpu
data seeds: 20, 21, 22
split seeds: 201, 203, 207
n states: 128
k actions: 32
grid size: 64
horizon: 4
channels: rgb, range, local, edge, noisy_rgb, gray_rgb, blur_rgb
representation: pca_probe_only
PCA dim: 32
normalization: probe_action_type_apply
k: 10
jitter: 0
runs: 9/9 complete
```

`edge` is RGB-derived diagnostic-only.  `local` is action-site-conditioned
diagnostic-only.  Redundancy controls are `noisy_rgb`, `gray_rgb`, and
`blur_rgb`.

## Main Pair Summary

| pair | action effect | static | shuffled | action - static | residualized action | residualized - shuffled | CKNNA action | CKNNA static | CKNNA residualized | conditioned bins positive | candidate screen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| range:local | +0.0319 | +0.0195 | +0.0044 | +0.0124 | +0.0236 | +0.0182 | +0.0191 | +0.0261 | +0.0220 | yes | no |
| rgb:blur_rgb | +0.1409 | +0.8077 | +0.0088 | -0.6668 | +0.1689 | +0.1570 | +0.1315 | +0.8799 | +0.1707 | yes | no |
| rgb:edge | +0.1191 | +0.2624 | +0.0086 | -0.1433 | +0.1282 | +0.1147 | +0.1090 | +0.2860 | +0.1187 | yes | no |
| rgb:gray_rgb | +0.2305 | +0.6504 | +0.0072 | -0.4200 | +0.2548 | +0.2393 | +0.2121 | +0.7193 | +0.2427 | yes | no |
| rgb:local | +0.1174 | +0.0747 | -0.0005 | +0.0426 | +0.1087 | +0.1009 | +0.0753 | +0.1020 | +0.0628 | yes | yes |
| rgb:noisy_rgb | +0.4637 | +0.8586 | +0.0154 | -0.3949 | +0.4818 | +0.4629 | +0.4773 | +0.9287 | +0.4990 | yes | no |
| rgb:range | +0.0232 | +0.0916 | +0.0022 | -0.0684 | +0.0247 | +0.0299 | +0.0185 | +0.0994 | +0.0273 | yes | no |

## Static-Conditioned Bins

All reported pairs have positive action-effect alignment above shuffled in all
four static-similarity bins.

```text
rgb:range conditioned adjusted means:
  lowest   +0.0648
  low_mid  +0.0631
  high_mid +0.0569
  highest  +0.0536

rgb:local conditioned adjusted means:
  lowest   +0.1750
  low_mid  +0.1689
  high_mid +0.1673
  highest  +0.1445
```

## Interpretation

This run strengthens the D-prime interpretation:

```text
static alignment captures shared scene layout;
action-effect alignment captures additional intervention-specific structure
that survives static controls.
```

It does not support the full-strength Framing D claim as a primary result.  For
the clean `rgb:range` pair, raw static remains stronger than action-effect:

```text
rgb:range action-effect: +0.0232
rgb:range static:        +0.0916
```

However, `rgb:range` remains non-null after static controls:

```text
rgb:range residualized action-effect:      +0.0247
rgb:range residualized minus shuffled:     +0.0299
rgb:range residualized CKNNA:              +0.0273
rgb:range static-conditioned bins:         all positive
```

The only exploratory cell satisfying the candidate screen is `rgb:local`.
Because `local` is action-site-conditioned, this is diagnostic evidence only and
does not open AERA implementation.

## Decision

```text
Framing D full strength: not established.
Framing D-prime: remains viable and stronger than after B6R.
AERA implementation: still blocked.
Next allowed step: AERA novelty survey and, if survey supports novelty, a
future preregistered replication of a frozen diagnostic cell.
Path-Building / Sand-Pushing audit: still blocked.
Custom simulator: blocked as main evidence.
```

Primary outputs:

```text
outputs/static_weak_realpw_v2_cpu/static_weak_pair_summary.csv
outputs/static_weak_realpw_v2_cpu/stageb6_knn_sensitivity.csv
outputs/static_weak_realpw_v2_cpu/static_controls_static_weak_v1/stageb6_static_residualized_knn.csv
outputs/static_weak_realpw_v2_cpu/static_controls_static_weak_v1/stageb6_static_conditioned_knn.csv
outputs/static_weak_realpw_v2_cpu/static_controls_static_weak_v1/stageb6_static_control_literature_metrics.csv
```


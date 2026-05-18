# Stage B.5 Preregistration

Stage B.5 replaces the over-strong B.4 primary gate with a held-out
same-action-set cross-channel gate.

B.5 tests held-out same-action-set alignment only. It does not claim
action-subset transfer and does not unlock Stage C by itself.

## Motivation

B.4 showed that state neighborhoods induced by disjoint action subsets are not
stable enough for primary claims. This is not necessarily a bug: local action
subsets can reveal different aspects of the same state.

The Stage B primary question is narrower:

> Given the same held-out action set, do action-effect signatures align across
> channels better than static signatures or shuffled-action controls?

## Gate Order

0. **Gate -1: setup validity**
   Exclude diagnostic channels, require probe-trained encoders for primary PCA,
   use non-transductive primary normalizations, report valid query counts, and
   record feature tie diagnostics. Same-channel probe-to-heldout remains a B.4
   diagnostic and is not a B.5 primary gate.
1. **Gate 0: held-out same-action-set redundancy calibration**
   Compare `D_rgb^heldout(s)` to `D_noisy_rgb^heldout(s)` and
   `D_gray_rgb^heldout(s)`. `rgb-blur_rgb` is reported as a broader low-pass
   diagnostic.
2. **Gate 1: `rgb-range` held-out action-effect > static / shuffled**
   Compare `D_rgb^heldout(s)` and `D_range^heldout(s)` against static held-out
   signatures and action-column shuffled held-out signatures.
3. **Gate 2: continuous observability**
   Test whether threshold-free joint detectability explains query-level
   action-effect overlap.

## Primary Scores

Primary continuous observability:

```text
detect_geom_rank_mean = mean_a sqrt(rank(d_rgb(s,a)) * rank(d_range(s,a)))
```

Secondary diagnostic:

```text
regular_minus_blind = regular_both_fraction - blind_either_fraction
```

The primary score is threshold-free; binary strata are not the main B.5 claim.

## Normalization Policy

Primary normalizations are:

- `none`
- `probe_global_apply`
- `probe_action_type_apply`

The B.5 v1 report must show all primary normalizations. A result is not a
primary pass if it only succeeds under a post-hoc best normalization. Diagnostic
transductive modes are excluded from the pass decision.

## B.2 Signal Branch

B.2 v1 observed `rgb-range` held-out all chance-adjusted overlap of `+0.0400`.
B.5 must report the comparable value and branch before any framing update:

- strengthened: B.5 value >= `+0.0600`
- replicated weak positive: `+0.0400 <= B.5 value < +0.0600`
- attenuated partial: `0 < B.5 value < +0.0400`
- disappeared: B.5 value <= `0`

If the signal disappears, the B.2 positive signal should not be used as support.

## Feature Diagnostics

B.5 reports:

- `pca_probe_only` primary representation;
- `raw_delta` diagnostic representation;
- `random_projection` diagnostic representation;
- optional `pca_all_action` diagnostic upper bound when an all-action model is
  provided.

HOSVD/tensor factorization is deferred until raw/random/all-action PCA identify
whether the failure is representation compression, encoder split, or signature
definition.

## Stage C Condition

Stage C remains blocked unless:

1. core redundancy controls pass under held-out same-action-set calibration;
2. `rgb-range` action-effect held-out exceeds static and shuffled controls;
3. continuous observability explains some query-level overlap or a stable
   action-coupling signal remains after controls.

Query-bootstrap intervals do not replace multi-seed / multi-split evidence.

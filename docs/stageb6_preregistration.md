# Stage B.6 Preregistration

Stage B.6 is a diagnostic hardening step for the Stage B.5 signal. It is not
a Stage C / PSP-like experiment and does not by itself unlock world-model
training.

## Purpose

Stage B.5 found a weak `rgb-range` held-out same-action-set signal under
`pca_probe_only / probe_action_type_apply`, but the result was PCA- and
normalization-dependent and the PCA features were tie-heavy. Stage B.6 asks:

> Is the B.5 `rgb-range` signal stable under k, jitter, PCA basis, PCA
> dimension, and alternative measurement primitives?

This is an artifact test, not a search for a better reporting cell.

## Primary Replication Cell

The pre-committed replication cell is:

```text
representation = pca_probe_only
normalization  = probe_action_type_apply
components     = 32
k              = 10
jitter         = 0
```

Passing this cell alone is not a full Stage B.6 pass. It only reproduces the
B.5 v1 condition.

## Robustness Families

Report full tables for:

- `k in {5, 10, 20}`
- PCA components `d in {16, 32, 64, 128}`
- jitter epsilon `0, 1e-6, 1e-5, 1e-4, 1e-3`
- representations:
  - `pca_probe_only` as primary-eligible;
  - `pca_all_action` as a transductive diagnostic upper bound only;
  - `raw_delta` and `random_projection` as diagnostic controls.

Do not select the best k, dimension, normalization, or representation as the
main result after seeing the grid.

## Primary Metrics

For kNN:

```text
chance_adjusted_overlap(rgb, range)
action_effect_minus_static paired bootstrap CI
action_effect_minus_shuffled paired bootstrap CI
```

All differences use the same query states and paired query bootstrap.

For continuous observability:

- primary: threshold-free `detect_geom_rank_mean`;
- secondary: `regular_minus_blind` and related threshold-derived scores.

For measurement-primitive sanity:

- state-flat linear CKA with row-permutation null;
- state-flat RSA Spearman with row-permutation null;
- bidirectional ridge transfer with row-permutation null;
- action-conditioned RSA averaged over held-out actions.

Raw CKA/RSA/probe values are not interpreted without their null calibration.

## Pass Labels

Use these labels after aggregation:

```text
full diagnostic pass
partial robust positive
PCA-dependent weak positive
tie-sensitive artifact
metric unstable
negative
```

Minimum requirements for `partial robust positive`:

- held-out redundancy controls remain positive in the primary replication cell;
- `rgb-range` primary cell has positive chance-adjusted overlap;
- action-effect beats static and shuffled in the paired point estimates;
- at least 2/3 k values are positive;
- at least 3/4 PCA dimensions are positive for `pca_probe_only`;
- jitter up to `1e-4` does not flip the sign in more than 30% of jitter seeds;
- `detect_geom_rank_mean` has positive direction.

`pca_all_action` is never counted for primary pass/fail. It only diagnoses
whether the probe-only PCA split is overly strict.

## Loopholes

B.6 results should not be used as primary evidence if any of the following
occur:

- all-action PCA is mixed into the primary gate;
- the held-out action set differs between static, action-effect, or shuffled
  comparisons;
- a result exists only at `k=10` or only at `d=32`;
- jitter destroys the redundancy controls or the `rgb-range` sign;
- measurement-primitive sanity is positive only before permutation calibration;
- the report highlights only `pca_probe_only / probe_action_type_apply` while
  hiding the rest of the grid.

## Literature Constraints

Stage B.6 is motivated by recent PRH critiques: cross-modal kNN alignment can
be fragile at scale and in many-to-many settings, raw similarity metrics need
permutation-null calibration, and useful alignment may live in a low-dimensional
subspace. Therefore, B.6 includes both local-neighborhood kNN diagnostics and
calibrated CKA/RSA/linear-transfer sanity checks.

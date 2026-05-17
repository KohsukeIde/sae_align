# Metrics

## Physical effect magnitude

The starter implementation uses changed-cell ratio:

```math
E_x(s,a) = \frac{1}{HW}\sum_{i,j} 1[x^a_{t+H,i,j} \ne x^{noop}_{t+H,i,j}].
```

## Channel detectability

For each observation channel, detectability is the mean absolute difference between do-action and no-op observations:

```math
d_m(s,a) = \mathrm{mean}(|o_m(x^a_{t+H}) - o_m(x^{noop}_{t+H})|).
```

Because different channels have different units, thresholds are set using within-channel quantiles.

## Blind-locus overlap

For channel `m`, the blind locus is:

```math
\Sigma_m = \{(s,a): E_x(s,a) \ge \epsilon_x,\ d_m(s,a) < \tau_m\}.
```

Pairwise Jaccard:

```math
J(\Sigma_m, \Sigma_n) = \frac{|\Sigma_m \cap \Sigma_n|}{|\Sigma_m \cup \Sigma_n|}.
```

Complementarity:

```math
C(m,n) = 1 - J(\Sigma_m, \Sigma_n).
```

## Redundancy probe

We ask whether one channel's action-effect response can predict another channel's response:

```math
\Delta o_n \leftarrow h(\Delta o_m).
```

The starter code uses random projections and ridge regression, reporting test-set R².  This is a diagnostic, not a final representation-learning result.


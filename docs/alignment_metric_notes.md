# Alignment Metric Notes

Date: 2026-05-19

These notes record the literature-method review used to design the Stage B.6
diagnostics. They are not experimental results. They explain why Stage B.6 does
not rely on a single kNN number, and why CKA/RSA/ridge and PCA-subspace
diagnostics were added.

## Review Protocol

The literature review was assigned as a fixed-paper method extraction task, not
as open-ended search. The reviewed papers were specified in advance:

- arXiv `2604.18572`
- arXiv `2602.14486`
- arXiv `2503.05283`

The subagent task was to inspect the local PDFs/text extracts and the current
repo implementation, then report:

- the alignment primitive used by each paper;
- how each paper's kNN or local-similarity variant differs;
- which parts are already implemented in Stage B.6;
- which loopholes remain.

No subagent was asked to use search as a discovery engine for additional
papers. Additional literature should be added through an explicit fixed-paper
review task or through references that are deliberately promoted into scope.

## Subagent Work Products

The fixed-paper review and implementation audit were split into separate
high-load tasks:

- literature-method extraction: identify the exact neighborhood, graph, and
  subspace alignment primitives in `2604.18572`, `2602.14486`, and
  `2503.05283`;
- implementation design: decide which primitives can be added without changing
  the preregistered B.6 primary gate;
- code review: check that new diagnostics cannot contaminate primary
  summaries;
- loophole audit: list circularity, split leakage, transductive subspace, and
  stale/incomplete-grid risks.

The resulting implementation keeps the literature-derived metrics separate from
the primary B.6 CSVs. Cycle-kNN, CKNNA, CCA, and SVCCA are written to
`b6_literature_metrics.csv`, and the primary summary remains based on the
precommitted heldout-to-heldout kNN grid.

## Papers Checked

### Back into Plato's Cave

Reference: <https://arxiv.org/abs/2604.18572>

This paper re-examines PRH-style cross-modal convergence at scale. The central
alignment metric is mutual kNN:

- L2-normalize the two representation spaces.
- For each shared datapoint, retrieve the `k` nearest neighbors independently
  in each modality.
- Score the fraction of overlapping neighbor identities.
- Average this per-sample overlap over the query set.

Important constraints for this repo:

- Raw mutual kNN is sensitive to gallery density. A fixed small `k` becomes
  stricter as the candidate gallery becomes denser.
- `k=10` was used in the original PRH-style setup, but the critique explicitly
  checks sensitivity to `k`.
- Mutual kNN can understate alignment in many-to-many regimes: two modalities
  may retrieve semantically good but different neighbors.
- Low mutual kNN is therefore not automatically a proof that the underlying
  representations are bad. It may mean the metric is asking for exact neighbor
  identity agreement.

Mapping to this repo:

- Stage B.6 keeps kNN overlap, but treats it as one local-topology diagnostic,
  not as the sole alignment evidence.
- B.6 includes `k in {5, 10, 20}` to avoid a single `k=10` claim.
- B.4's failed probe-to-heldout split-half gate is recorded as an overly strong
  action-subset-transfer requirement, not as proof that action-effect alignment
  is absent.
- B.6 heldout-to-heldout kNN is closer to the required claim: same held-out
  action set, cross-channel neighborhood overlap.

### Revisiting the Platonic Representation Hypothesis

Reference: <https://arxiv.org/abs/2602.14486>

This paper argues that raw representational-similarity metrics can be confounded
by model scale and representation dimensionality. It evaluates a metric family,
including:

- linear CKA and kernel CKA;
- RSA via Spearman correlation of dissimilarity matrices;
- CCA, SVCCA, PWCCA, RV coefficient, and Procrustes-style metrics;
- mutual kNN (`mKNN`);
- cycle-kNN, which requires bidirectional neighborhood consistency;
- CKNNA, which applies a CKA-style comparison to kNN graph adjacency.

The key methodological constraint is calibration:

- raw scores are compared against a permutation-null baseline;
- local neighborhood metrics can retain alignment after calibration even when
  global spectral metrics weaken;
- mKNN measures topological agreement: which samples are neighbors;
- small-bandwidth CKA-RBF measures local metric agreement: how close the
  neighbors are.

Important constraints for this repo:

- A raw global CKA/RSA/probe value is not enough; it needs a null or shuffled
  calibration.
- kNN and CKA/RSA answer different questions. kNN asks about neighbor identity,
  while CKA/RSA ask about broader similarity geometry.
- A positive CKA/RSA result cannot automatically replace kNN evidence, and a
  weak kNN result does not automatically kill all alignment evidence.

Mapping to this repo:

- Stage B.6 adds calibrated measurement-primitive sanity checks:
  - state-flat linear CKA with row-permutation null;
  - state-flat RSA with row-permutation null;
  - bidirectional ridge transfer with row-permutation null;
  - action-conditioned RSA over held-out actions.
- B.6 interprets these only as sanity diagnostics. They are not promoted to a
  final claim unless the kNN and control evidence also behave sensibly.
- After B.6 v1, diagnostic-only cycle-kNN and CKNNA rows were added in
  `b6_literature_metrics.csv`. They are not part of the preregistered primary
  gate.

### Escaping Plato's Cave

Reference: <https://arxiv.org/abs/2503.05283>

This paper studies post-training alignment of 3D and text latent spaces. Its
main lesson for this repo is not a new kNN definition, but the importance of
subspace selection:

- naive full-space 3D-text alignment is weak;
- CCA is used to identify lower-dimensional correlated subspaces;
- affine translation and local CKA are then applied in the projected subspace;
- matching and top-k retrieval improve after operating in the selected
  subspace;
- local CKA is used for query-level retrieval/matching with anchor sets.

Important constraints for this repo:

- Weak raw-delta alignment does not by itself imply absence of shared structure;
  the useful structure may live in a lower-dimensional subspace.
- Conversely, any subspace method can become post-hoc if it is selected using
  evaluation data.
- Transductive subspace diagnostics must be clearly separated from primary
  evidence.

Mapping to this repo:

- Stage B.6 includes `pca_probe_only` as the primary-eligible compressed
  representation.
- Stage B.6 includes `pca_all_action` only as a transductive diagnostic upper
  bound, never as primary evidence.
- Stage B.6 includes `raw_delta` and `random_projection` as controls to check
  whether the PCA signal is a denoising/subspace effect or an artifact.
- CCA and SVCCA are now implemented as diagnostic-only rows in
  `b6_literature_metrics.csv`. They are not primary evidence because subspace
  metrics can stay positive under some controls and can become circular if
  fitted on evaluation data.

## Current Loopholes

- Raw kNN overlap can be too strict for many-to-many or dense-gallery regimes.
- A single `k` can create a fragile conclusion.
- Ties or duplicate signatures can make kNN unstable.
- Global CKA/RSA/probe scores can be inflated without permutation calibration.
- Subspace selection can become circular if all-action or evaluation data are
  used as primary evidence.
- PCA-compressed positives with weak raw-delta support should be interpreted as
  possible denoising/subspace evidence, not yet as representation-independent
  proof.

## kNN And Local-Metric Differences

The three reviewed papers do not use the same neighborhood/alignment primitive.
This matters because Stage B results should not be described as if all PRH
metrics were interchangeable.

| Source | Primitive | What is compared | Main risk |
| --- | --- | --- | --- |
| `2604.18572` | raw mutual kNN overlap | overlap of nearest-neighbor identities in two modalities after normalization | sensitive to gallery density, k, and one-to-one assumptions |
| `2602.14486` | calibrated mKNN, cycle-kNN, CKNNA, CKA/RSA/CCA family | neighborhood identity, bidirectional consistency, kNN graph adjacency, or global similarity geometry under permutation nulls | raw scores can be inflated; each metric answers a different topology/geometry question |
| `2503.05283` | CCA-selected subspace plus affine or local CKA retrieval/matching | query/anchor alignment in a correlated low-dimensional subspace | subspace selection can become circular if evaluation data enter the primary fit |

Stage B.6 currently implements only a subset:

- implemented: heldout-to-heldout kNN overlap, k-sweep, jitter, PCA component
  sweep, raw/random diagnostics, calibrated state-flat CKA/RSA/ridge, and
  action-conditioned RSA;
- implemented as diagnostic-only additions after B.6 v1: cycle-kNN, CKNNA,
  CCA, and SVCCA, written separately to `b6_literature_metrics.csv`;
- not implemented: PWCCA, local CKA retrieval, or retrieval/matching metrics.

Therefore, Stage B.6 should be described as a kNN-plus-sanity diagnostic, not
as a full reproduction of the PRH metric suite or the 3D-text subspace
alignment pipeline.

## Current Repo Commitments

- Use heldout-to-heldout same-action-set kNN as the primary Stage B.5/B.6
  alignment setup.
- Keep k-sweep, jitter, and PCA-dimension sweep in the B.6 diagnostic grid.
- Keep calibrated CKA/RSA/ridge as measurement sanity checks, not final claims.
- Keep cycle-kNN, CKNNA, CCA, and SVCCA diagnostic-only unless a later
  preregistration promotes them.
- Treat `pca_all_action` as diagnostic only.
- Treat continuous observability as the current working framing; binary
  regular/blind strata are secondary until stronger evidence appears.

## Future Metric Additions

If Stage B remains weak but nonzero, the next method additions should be:

- PWCCA diagnostics for shared action-effect subspaces;
- retrieval-style evaluations over action-effect signatures, inspired by the
  3D-text matching/retrieval setup, but with precommitted action/state splits.

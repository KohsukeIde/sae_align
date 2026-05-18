# How to Extend This Repo

## Add Stage A.5: deployable proxy agreement

Create:

```text
scripts/analyze_proxy_agreement.py
src/sae_align/analysis/proxies.py
```

Recommended proxies:

- observation-delta proxy;
- action-sensitivity proxy;
- multi-sensor consensus proxy;
- learned blindness classifier.

Report AUROC, AUPRC, IoU/Jaccard, best-F1, and threshold curves against oracle strata.

## Extend Stage B: regular/blind pairwise and stratified action-effect kNN

The current NumPy Stage B pilot already provides:

```text
scripts/train_transition_encoder.py
scripts/analyze_stratified_knn.py
src/sae_align/models/encoders.py
src/sae_align/analysis/knn_alignment.py
```

Primary Stage B evidence is the regular/blind pairwise alignment report over default trainable channels. Keep `event_response` excluded because it is a post-action diagnostic response channel. Only include it in separately labeled leakage diagnostics with `--allow-leakage-diagnostic`.

Required controls:

- action-only controls;
- shuffled-strata controls;
- shuffled-action controls.
- same-action-type and same-action-id restricted kNN;
- action-residualized kNN as a sensitivity control;
- static-vs-action-effect kNN comparison.

Main control: strata must be defined using `x_star` and `o_m`, not representation deltas. Current Stage 0/B pilots are NumPy-only and do not require qsub, a GPU, or CUDA.

Stage B.1 is the immediate hardening step before Stage C. Use
`scripts/run_stageb_b1_smoke.sh` for a CPU smoke run. Read restricted-kNN results
through `chance_adjusted_overlap`, `n_valid_queries`, and `mean_effective_k`.
Static comparisons should exclude same-state neighbors when `state_id` is
available, because the same state appears once per action.

## Add Stage C: selective prediction

Create:

```text
scripts/train_world_model.py
scripts/evaluate_world_model.py
src/sae_align/models/world_model.py
src/sae_align/training/loss_weights.py
```

Minimal baselines:

- uniform;
- change-mask;
- PSP-like;
- proxy strata;
- PSP-like + proxy strata;
- oracle strata;
- random mask.

## Add external benchmark

Do this only after Stage A/B/C signals are positive. Start with an offline Distracting Control Suite subset and proxy-only weighting.

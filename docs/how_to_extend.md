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

Stage B.1 is the immediate hardening step before Stage B.2. Use
`scripts/run_stageb_b1_smoke.sh` for a CPU smoke run. Read restricted-kNN results
through `chance_adjusted_overlap`, `n_valid_queries`, and `mean_effective_k`.
Static comparisons should exclude same-state neighbors when `state_id` is
available, because the same state appears once per action.

Stage B.2 is the current hardening priority: state-level action-effect signature alignment.
For each state `s`, estimate a state-level signature `D_m(s)` from probe actions
and compare signatures across channels on held-out test actions. Define
`regular_state` / `blind_state` strata at the state level, rather than treating
each state-action row as an independent stratum member. The probe/test action
split is required so the signature used to choose or describe a state is not
evaluated on the same actions.

Use `--dense-sampling full-states` when generating Stage 0 data for Stage B.2;
random dense subsets are insufficient because `D_m(s)` requires all action
columns for each retained state. The smoke command is:

```bash
PYTHONPATH=src bash scripts/run_stageb_b2_smoke.sh outputs/stageb_b2_smoke
```

For primary held-out action claims, train the action-effect encoder with the
same probe action IDs used by `scripts/analyze_state_signature_knn.py`. The
script fails by default when the encoder was trained on all actions; override
that only for diagnostics with `--allow-all-action-trained-model`. It also
checks the Stage 0 data fingerprint stored in the encoder metadata; bypass this
only for diagnostics with `--allow-cross-data-model`.
Use `--bootstrap-repeats` for non-smoke Stage B.2 runs so
`state_signature_bootstrap_ci.csv` records query-bootstrap confidence intervals.
Use `state_strata_fraction_summary.csv` to choose a stable stratum policy before
interpreting `regular_state` versus `blind_state`.

The first larger local CPU Stage B.2 run is recorded in
`docs/stageb2_v1_cpu_experiment.md`. Treat it as a real experiment with a
negative/weak decision, not as a pass. Before moving to Stage C, repair the
state-stratum policy, make probe-to-heldout transfer the primary pass metric,
and require redundancy controls to pass that cross-action transfer check.

Stage B.3 formalizes that repair as a framing-decision gate. Read
`docs/stageb3_preregistration.md` first, then use:

```bash
PYTHONPATH=src bash scripts/run_stageb_b3_smoke.sh outputs/stageb_b3_smoke
```

The B.3 reports test balanced action splits, non-transductive normalization,
held-out static/shuffled controls, redundancy probe-to-heldout calibration, and
continuous observability scores.

## Add Stage C: selective prediction

Stage C is still too early. Add selective-prediction baselines only after Stage
B.1 controls and Stage B.2 state-level action-effect signature alignment are
stable.

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

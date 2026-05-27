# Stratified Action-Effect Alignment (starter repo)

This repository is a **starter codebase** for the first experimental phase of the ICLR project:

> **Stratified Action-Effect Alignment for Selective World Model Prediction**

The goal of this first phase is *not* to train a full world model or reproduce Dreamer/PSP.  The goal is to verify the core experimental premise:

1. We can construct non-redundant observation channels with distinct action-effect blind loci.
2. These blind loci are stable under action-subset and threshold sweeps.
3. Primary redundancy controls such as `noisy_rgb`, `gray_rgb`, and `blur_rgb` behave as redundant channels.
4. Derived diagnostics such as edge maps are interpreted separately from redundancy controls.
5. We can later use these strata for stratified action-effect alignment and selective prediction.

The repository is designed to be extended incrementally. Stage 0 is implemented, and the NumPy Stage B action-effect kNN pipeline is available for pilot diagnostics. Stage B.6 is a partial robust positive: the metric is no longer obviously broken, but the effect size is still small and binary strata remain weak. Stage C0/C0.5 were No-go: weighted-ridge and logistic detector smokes did not produce a strong oracle-positive behavioral detector. Stage D0' was also No-go: predictor-grounded observability produced a raw AUPRC lift, but oracle-event failed the hard gate. Stage D0 real-Powderworld oracle-positive detector audit was also No-go: generator sanity passed, but oracle-event failed the hard gate. The follow-up Action-IV Step-2 official-task oracle sanity audit on real Powderworld DestroyAll also failed (`oracle_mean_auprc_delta=+0.00168`). A one-shot postmortem then showed that the DestroyAll target is learnable (`uniform obs+action AUPRC=0.9064`, prevalence `0.2281`) and oracle-as-feature reaches AUPRC `1.0`; therefore the failed gate is best read as an invalid oracle sample-weighting upper-bound, not as an unlearnable target. Real-Powderworld B6 static controls weakened the static-artifact concern, but the preregistered B6R replication is a strict No-go / near-miss (`static_residualized_probefit=+0.0293`, threshold `+0.03`). Full Stage C / PSP-like / Dreamer-like comparison, Action-IV Step 3, and AERA implementation remain blocked.

---

## Why this repo exists

Current representation-alignment work often asks whether two models or modalities globally share similar representation geometry.  This project asks a more controlled question:

> Before asking whether representations align, can the observation channel even observe the action-induced physical change?

We therefore separate action effects into:

- **physical null**: the action did not meaningfully change the underlying world state;
- **modality blind**: the world changed, but the observation channel did not register it;
- **regular**: the world changed and the observation channel registered it.

Representation blind regions are intentionally **not** part of the main Stage 0 strata. They are reserved for later diagnostic experiments to avoid circularity between strata definition and representation-alignment metrics.

---

## Current scope

Implemented now:

- a lightweight toy Powderworld-like simulator with deterministic local interactions;
- a fixed observation-channel specification:
  - RGB;
  - range-like channel;
  - local interaction channel;
  - `event_response` as a diagnostic-only post-action response channel;
  - semantic grid as privileged control;
  - edge map as derived diagnostic channel;
  - `noisy_rgb`, `gray_rgb`, and `blur_rgb` as redundancy controls;
- no-op vs do-action counterfactual rollouts;
- effect-magnitude and detectability computation;
- blind-locus overlap and complementarity matrices;
- redundancy probes from one channel's action-effect response to another;
- K-sweep and threshold-sweep support;
- stage reports and plots;
- NumPy Stage B transition encoders and regular/blind pairwise kNN alignment reports.
- Stage B.1 confound-control smoke reports for action-restricted, action-residualized, and static-vs-action-effect comparisons;
- Stage B.2 state-level action-effect signature smoke/pilot reports with full-state dense sampling, probe-action encoder training, held-out action scoring, and static baselines.
- Stage B.3/B.4 calibration reports for redundancy transfer, normalization, same-channel split-half reliability, and feature tie diagnostics.
- Stage B.5/B.6 held-out same-action-set reports, artifact diagnostics, and literature-derived diagnostic metrics.

Not implemented yet:

- PSP-like or Dreamer-like baselines;
- world-model training;
- Distracting Control Suite external validation.

---

## Installation

```bash
conda create -n sae-align python=3.10 -y
conda activate sae-align
pip install -e .
```

Or with pip only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Quickstart

Run the Stage 0 smoke test:

```bash
bash scripts/run_stage0_smoke.sh
```

This writes:

```text
outputs/stage0_smoke/
  data/stage0_dataset.npz
  reports/stage0_summary.json
  reports/blind_jaccard.csv
  reports/complementarity.csv
  reports/redundancy_r2.csv
  figures/effect_histograms.png
  figures/blind_jaccard.png
  figures/complementarity.png
  figures/redundancy_r2.png
  figures/examples.png
```

For a larger run:

```bash
python scripts/make_stage0_dataset.py \
  --config configs/stage0_toy.json \
  --out outputs/stage0_main/data/stage0_dataset.npz \
  --n-states 1000 \
  --k-actions 64 \
  --grid-size 32 \
  --seed 0

python scripts/analyze_stage0_channels.py \
  --data outputs/stage0_main/data/stage0_dataset.npz \
  --out outputs/stage0_main \
  --k-sweep 16 32 64 \
  --threshold-quantiles 0.05 0.10 0.20
```

---

## Repository layout

```text
sae_align_repo/
  configs/                  # JSON configs for staged experiments
  docs/                     # protocol and design-freeze documents
  scripts/                  # executable scripts
  src/sae_align/
    envs/                   # toy simulator + Powderworld adapter stub
    analysis/               # stage-0 metrics and plotting
    utils/                  # io, seeding, misc helpers
  tests/                    # minimal sanity tests
```

This follows the common pattern of compact ML research repos: a top-level README with minimal commands, scripts/configs for reproducing figures, and docs for protocol details.

---

## Stage 0 Decision Criteria

Proceed to representation-alignment experiments only if:

1. trainable sensor-like channels such as `rgb`, `range`, and `local` have distinct blind loci;
2. RGB plus the defined primary redundancy controls, `noisy_rgb`, `gray_rgb`, and `blur_rgb`, behaves as low-complementarity / high-redundancy controls;
3. blind-locus complementarity is stable under K-sweep and threshold-sweep;
4. the redundancy probe predicts RGB-derived controls much better than non-redundant channels;
5. `event_response` is reported as a diagnostic-only post-action response and is excluded from default Stage B trainable channels.

Do not require edge to have high blind overlap with RGB. Edge is a derived diagnostic that can expose boundary-change sensitivity distinct from RGB. If the Stage 0 criteria fail, do not proceed to Stage B selective world-model prediction. Redesign the observation channels first.

---

## Notes

This starter repo uses a toy simulator so that Stage 0 can be run without installing the original Powderworld code.  A `PowderworldAdapter` stub is provided in `src/sae_align/envs/powderworld_adapter.py` for later integration.

## Stage B Pilot

After Stage 0, run the current NumPy Stage B pilot:

```bash
python scripts/train_transition_encoder.py \
  --data outputs/stage0_v1/data/stage0_dataset.npz \
  --out outputs/stageb_v1

python scripts/analyze_stratified_knn.py \
  --data outputs/stage0_v1/data/stage0_dataset.npz \
  --model outputs/stageb_v1/transition_encoders.npz \
  --out outputs/stageb_v1_knn
```

The primary Stage B report is the regular/blind pairwise alignment report over default trainable channels such as `rgb`, `range`, and `local`. It uses the Stage 0 oracle strata and reports pairwise overlap by regular/blind strata.

Required scientific guards for Stage B pilots:

1. Keep `event_response` out of the primary Stage B channel set. It is diagnostic-only because it is a post-action response.
2. Use `--allow-leakage-diagnostic` only for a separately labeled leakage diagnostic, never as primary Stage B evidence.
3. Include action-only and shuffled controls, including shuffled-strata and shuffled-action controls, when interpreting Stage B results.
4. Do not treat `edge` as the RGB redundancy control. The primary RGB redundancy controls are `noisy_rgb`, `gray_rgb`, and `blur_rgb`.

For the current Stage B.1 confound-control smoke run:

```bash
PYTHONPATH=src bash scripts/run_stageb_b1_smoke.sh outputs/stageb_b1_smoke
```

This adds same-action-type, same-action-id, action-residualized, and
static-vs-action-effect reports. See `docs/stageb_status.md` for the current
interpretation and remaining Stage B loopholes.

The current priority is no longer PSP/Dreamer. B.6 remains weak positive
alignment evidence, but C0/C0.5/D0' did not convert that signal into a strong
behavioral validation. ToyPowderWorld Stage C is stopped. D0 real-Powderworld audit is complete and No-go; it is not a detector-tweak loop. B6R real-Powderworld static-control replication is also a strict No-go / near-miss, so AERA implementation remains blocked while the contribution survey is handled separately. See
`docs/stagec0_v1_cpu_experiment.md`, `docs/stagec05_precommit.md`,
`docs/stagec05_v1_cpu_experiment.md`, `docs/staged0p_precommit.md`, and
`docs/staged0p_v1_cpu_experiment.md`, plus `docs/staged0_precommit.md`, `docs/staged0_status.md`, and
`docs/staged0_v1_cpu_experiment.md`, as well as `docs/stageb6r_preregistration.md` and `docs/stageb6r_v1_cpu_experiment.md`.

For a minimal Stage B.2 smoke run:

```bash
PYTHONPATH=src bash scripts/run_stageb_b2_smoke.sh outputs/stageb_b2_smoke
```

This uses `--dense-sampling full-states` so each selected state has the full
action bank needed to form `D_m(s)`. It also trains the action-effect encoder
only on the probe action IDs before evaluating held-out action signatures.

A larger local CPU Stage B.2 v1 run is recorded in
`docs/stageb2_v1_cpu_experiment.md`. That run is an experiment, not only an
implementation check, and it does not pass the current Stage B.2 scientific
gate. Stage C remains blocked.

For the Stage B.3 framing-decision gate:

```bash
PYTHONPATH=src bash scripts/run_stageb_b3_smoke.sh outputs/stageb_b3_smoke
```

Read `docs/stageb3_preregistration.md` before interpreting the result. B.3 is
designed to decide between metric repair, continuous observability, and
action-coupling framing; it is not a Stage C experiment.

For the Stage B.4 split-half reliability gate:

```bash
PYTHONPATH=src bash scripts/run_stageb_b4_smoke.sh outputs/stageb_b4_smoke
```

Read `docs/stageb4_preregistration.md` and
`docs/stageb4_v1_cpu_experiment.md`. B.4 v1 did not pass: identity checks
passed, but same-channel split-half reliability and redundancy calibration
failed under CI.

For the Stage B.5 held-out same-action-set gate:

```bash
PYTHONPATH=src bash scripts/run_stageb_b5_smoke.sh outputs/stageb_b5_smoke
```

Read `docs/stageb5_preregistration.md` and
`docs/stageb5_v1_cpu_experiment.md`. B.5 v1 is partial positive, not a full
pass: `pca_probe_only / probe_action_type_apply` recovers a weak `rgb-range`
signal, but raw/random diagnostics are weaker and the PCA result is tie-heavy.
Stage C remains blocked.

For the Stage B.6 artifact and measurement-primitive diagnostics:

```bash
PYTHONPATH=src bash scripts/run_stageb_b6_v1_cpu.sh outputs/stageb_b6_v1_cpu
```

Read `docs/stageb6_preregistration.md` first. B.6 checks whether the B.5
signal survives k-sweep, tie-jitter, PCA component sweep, all-action PCA
diagnostics, and calibrated CKA/RSA/ridge sanity checks. This grid is larger
than the B.5 pilot; use `scripts/submit_stageb_b6_v1_cpu_array.sh` for the
CPU array version. GPU is not required.
The literature-method mapping behind these diagnostics is recorded in
`docs/alignment_metric_notes.md`.

The Stage B.6 v1 CPU array is recorded in
`docs/stageb6_v1_cpu_experiment.md`. It is a partial robust positive:
`rgb-range` remains positive in the preregistered primary cell, continuous
observability is positively associated with overlap, and raw/random diagnostics
are positive but weaker. Stage C is still not automatic.

For the B.6 primary-cell larger replication:

```bash
bash scripts/submit_stageb_b6_primary_cell_v1_cpu_array.sh
```

This submits an ABCI CPU array over `3 data seeds x 3 split seeds x 2 PCA
dims`. It keeps the literature-derived metrics diagnostic-only in
`b6_literature_metrics.csv`; they are not mixed into the existing B.6 primary
summary.

After the array finishes, aggregate with the expected-task guard:

```bash
PYTHONPATH=src python scripts/summarize_stageb6_grid.py \
  --root outputs/stageb_b6_primary_cell_v1_cpu \
  --out outputs/stageb_b6_primary_cell_v1_cpu \
  --expected-report-dirs 18
```

The completed run is recorded in
`docs/stageb6_primary_cell_v1_cpu_experiment.md`: the fixed `d=32/k=10`
primary cell stayed positive in `9/9` runs, with `rgb-range` adjusted mean
`+0.0275`, redundancy positives `27/27`, and CKNNA as the cleanest
diagnostic-only literature metric.

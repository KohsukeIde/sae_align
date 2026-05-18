# Stratified Action-Effect Alignment (starter repo)

This repository is a **starter codebase** for the first experimental phase of the ICLR project:

> **Stratified Action-Effect Alignment for Selective World Model Prediction**

The goal of this first phase is *not* to train a full world model or reproduce Dreamer/PSP.  The goal is to verify the core experimental premise:

1. We can construct non-redundant observation channels with distinct action-effect blind loci.
2. These blind loci are stable under action-subset and threshold sweeps.
3. Primary redundancy controls such as `noisy_rgb`, `gray_rgb`, and `blur_rgb` behave as redundant channels.
4. Derived diagnostics such as edge maps are interpreted separately from redundancy controls.
5. We can later use these strata for stratified action-effect alignment and selective prediction.

The repository is designed to be extended incrementally. Stage 0 is implemented, and the first NumPy Stage B action-effect kNN pipeline is available for pilot diagnostics. Current Stage 0/B pilots are NumPy-only and do not require qsub, a GPU, or CUDA.

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

# Stratified Action-Effect Alignment (starter repo)

This repository is a **starter codebase** for the first experimental phase of the ICLR project:

> **Stratified Action-Effect Alignment for Selective World Model Prediction**

The goal of this first phase is *not* to train a full world model or reproduce Dreamer/PSP.  The goal is to verify the core experimental premise:

1. We can construct non-redundant observation channels with distinct action-effect blind loci.
2. These blind loci are stable under action-subset and threshold sweeps.
3. Negative controls such as RGB-derived edge maps and noisy RGB copies behave as redundant channels.
4. We can later use these strata for stratified action-effect alignment and selective prediction.

The repository is designed to be extended incrementally.  Stage 0 and Stage A are implemented; later stages are scaffolded in the docs.

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
  - global event channel;
  - semantic grid as privileged control;
  - edge map as RGB-derived negative control;
  - noisy RGB copy as redundancy control;
- no-op vs do-action counterfactual rollouts;
- effect-magnitude and detectability computation;
- blind-locus overlap and complementarity matrices;
- redundancy probes from one channel's action-effect response to another;
- K-sweep and threshold-sweep support;
- stage reports and plots.

Not implemented yet:

- transition encoders;
- stratified action-effect kNN;
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

## Stage 0 decision criteria

Proceed to representation-alignment experiments only if:

1. the four main channels have distinct blind loci;
2. RGB+edge and RGB+noisy-RGB behave as low-complementarity controls;
3. blind-locus complementarity is stable under K-sweep and threshold-sweep;
4. the redundancy probe predicts RGB-derived controls much better than non-redundant channels;
5. the global event channel does not leak the exact downstream target.

If these fail, do not proceed to selective world-model prediction. Redesign the observation channels first.

---

## Notes

This starter repo uses a toy simulator so that Stage 0 can be run without installing the original Powderworld code.  A `PowderworldAdapter` stub is provided in `src/sae_align/envs/powderworld_adapter.py` for later integration.


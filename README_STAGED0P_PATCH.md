# Stage D0' Patch: Predictor-Grounded Observability Precommit + One-Shot Test

This patch adds a **single preregistered ToyPowderWorld follow-up** after Stage C0/C0.5 No-go.
It has been integrated and run as `outputs/staged0p_v1_cpu_v3`; the result is
No-go because the hard `oracle_event` gate failed.

The purpose is not to keep tweaking detectors. The purpose is to test one specific hypothesis:

> The current `detect_geom_rank_mean` observability score may be too predictor-utility-naive. A predictor-grounded score may connect the weak Stage B.6 action-effect alignment signal to behavioral prediction utility.

This patch adds:

- `docs/staged0p_precommit.md`: stop rules and branch decisions before running D0'.
- `docs/staged0_environment_migration_plan.md`: D0 migration gate if D0' fails or is inconclusive.
- `scripts/train_staged0p_predictor_grounded.py`: NumPy logistic detector with predictor-grounded weighting.
- `scripts/summarize_staged0p_grid.py`: aggregate seed/grid results and emit decision rows.
- `scripts/run_staged0p_smoke.sh`: quick local run on an existing Stage-0 dataset.
- `scripts/run_staged0p_v1_cpu.sh`: small CPU grid over data seeds.
- `scripts/submit_staged0p_v1_cpu_array.sh`: site-local PBS array template
  for larger reruns. The recorded v1 run used local CPU only.

## Apply

From repo root:

```bash
unzip sae_align_staged0p_patch.zip
rsync -av sae_align_staged0p_patch/ ./
```

## Smoke

Use an existing dense Stage-0 dataset, for example one produced under the B.6 primary-cell array:

```bash
PYTHONPATH=src bash scripts/run_staged0p_smoke.sh \
  outputs/staged0p_smoke \
  outputs/stageb_b6_primary_cell_v1_cpu/seed_0/split_17/pca_32/stage0/stage0_dataset.npz
```

## Aggregate

```bash
PYTHONPATH=src python scripts/summarize_staged0p_grid.py \
  --root outputs/staged0p_v1_cpu \
  --out outputs/staged0p_v1_cpu
```

## Decision

Run D0' once. The oracle-event gate is hard: if `oracle_event` does not exceed
uniform by the precommitted threshold, predictor-grounded positives are
diagnostic-only. The recorded v1 run failed this gate, so ToyPowderWorld Stage
C is stopped. Do not start C0.7/C0.8 detector tweaks.

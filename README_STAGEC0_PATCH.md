# Stage C0 Patch

This patch adds the first behavioral validation step after the Stage-B6 partial robust positive result.

## Apply

From the repo root:

```bash
unzip sae_align_stagec0_patch.zip
rsync -av sae_align_stagec0_patch/ ./
```

## Smoke

```bash
PYTHONPATH=src bash scripts/run_stagec0_smoke.sh outputs/stagec0_smoke
```

## V1 grid

```bash
PYTHONPATH=src bash scripts/run_stagec0_v1_cpu.sh outputs/stagec0_v1_cpu outputs/stage0_v1/data/stage0_dataset.npz
```

If the Stage-0 dataset lacks `obs0_<channel>` keys, regenerate it with `--store-static-obs`.

The ABCI array helper is PBS-style:

```bash
DATA=outputs/stage0_v1_static/data/stage0_dataset.npz \
OUT=outputs/stagec0_v1_cpu_array \
bash scripts/submit_stagec0_v1_cpu_array.sh
```

## Output

Main files:

```text
reports/stagec0_results.csv
reports/stagec0_summary.csv
reports/stagec0_decision_summary.json
stagec0_grid_summary.csv
stagec0_decision_grid.csv
stagec0_method_delta_summary.csv
```

## Scientific role

Stage C0 is not a full method result. It is a smoke test for whether continuous action-effect observability has behavioral utility. Do not start PSP/Dreamer baselines unless Stage C0 passes its preregistered go conditions.

The primary observability score is train-split geometric detectability over
`rgb` and `range`. Event/changed-cell F1 is scored from predicted magnitudes,
and diagnostic channels are rejected as inputs.

# Stage C0 Patch Notes

This patch adds Stage C0, a minimal behavioral smoke test.

## Added

- `scripts/train_stagec0_prediction.py`
- `scripts/summarize_stagec0_grid.py`
- `scripts/run_stagec0_smoke.sh`
- `scripts/run_stagec0_v1_cpu.sh`
- `scripts/submit_stagec0_v1_cpu_array.sh`
- `scripts/submit_stageb6_primary_cpu_array.sh`
- `docs/stagec0_preregistration.md`
- `docs/stagec0_status.md`
- `docs/stagec0_result_template.md`

## Design

Stage C0 deliberately avoids PSP/Dreamer comparisons. It tests whether the continuous action-effect observability score has behavioral utility in simple prediction heads.

The default comparison is:

```text
uniform / change_mask / observability / shuffled_observability / oracle_event
```

The result table explicitly compares Stage C0 effect sizes to the Stage-B6 kNN reference signal of ~0.05.

## Application

Unzip at repo root and run:

```bash
PYTHONPATH=src bash scripts/run_stagec0_smoke.sh outputs/stagec0_smoke
```

If the repo already has `outputs/stage0_v1/data/stage0_dataset.npz` with static observations:

```bash
PYTHONPATH=src bash scripts/run_stagec0_v1_cpu.sh outputs/stagec0_v1_cpu outputs/stage0_v1/data/stage0_dataset.npz
```

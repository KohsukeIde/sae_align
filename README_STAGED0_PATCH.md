# Stage D0 patch

This patch adds a new preregistered Stage D0: a real-Powderworld oracle-positive
detector audit.

## Included files

```text
src/sae_align/envs/powderworld_real_adapter.py
scripts/make_staged0_powderworld_dataset.py
scripts/compare_staged0_sanity.py
scripts/run_staged0_smoke_toy.sh
scripts/run_staged0_v1_cpu.sh
scripts/submit_staged0_cpu_array.sh
scripts/summarize_staged0p_grid.py
configs/staged0_powderworld.json
configs/staged0_toy_smoke.json
configs/staged0_datasets.example.txt
docs/staged0_precommit.md
docs/staged0_environment_integration.md
docs/staged0_status.md
docs/staged0_sanity_check.md
docs/staged0_v1_cpu_experiment.md
```

The patch reuses the existing `scripts/train_staged0p_predictor_grounded.py` and
`scripts/summarize_staged0p_grid.py` for the detector audit. It does not add PSP,
Dreamer, or RL training.

## Apply

```bash
unzip sae_align_staged0_patch.zip
rsync -av sae_align_staged0_patch/ ./
```

## Smoke test without original Powderworld

```bash
PYTHONPATH=src:. bash scripts/run_staged0_smoke_toy.sh outputs/staged0_smoke_toy
```

## Real Powderworld audit

Install optional upstream package first:

```bash
pip install powderworld
```

Then run:

```bash
PYTHONPATH=src:. bash scripts/run_staged0_v1_cpu.sh outputs/staged0_v1_cpu
```

D0 only justifies Stage C1 if both oracle and observability/predictor-grounded
gates pass with at least `+0.10` behavioral lift.

## Recorded v1 result

The integrated v1 run used qsub over three generator seeds and completed as
`1779238[].pbs1`. The decision was:

```text
branch_3_oracle_failed_stop_environment
```

See `docs/staged0_v1_cpu_experiment.md`. Stage C1 / PSP-like comparison remains
blocked.

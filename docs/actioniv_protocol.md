# Action-IV Protocol

## Phase logic

This phase starts after scalar observability weighting failed in ToyPowderWorld and real Powderworld detector audits.  The new method keeps the action-effect object structured instead of collapsing it into weights.

## Step 2: official task oracle sanity

Generate a task dataset:

```bash
PYTHONPATH=src:. python scripts/make_actioniv_task_dataset.py \
  --backend synthetic \
  --out outputs/actioniv_smoke/data/task_dataset.npz \
  --n-states 128 \
  --actions-per-state 8 \
  --seed 0
```

Train the oracle detector:

```bash
PYTHONPATH=src:. python scripts/train_actioniv_task_oracle.py \
  --data outputs/actioniv_smoke/data/task_dataset.npz \
  --out outputs/actioniv_smoke/task_oracle \
  --input-channels rgb \
  --methods uniform oracle_task change_mask observability shuffled_observability \
  --alphas 1 2 4 8 16 32
```

For real Powderworld, use:

```bash
PYTHONPATH=src:. bash scripts/run_actioniv_powderworld_destroy.sh outputs/actioniv_destroy
```

## Step 3: Action-IV effect prototype

```bash
PYTHONPATH=src:. python scripts/train_actioniv_effect_encoder.py \
  --data outputs/actioniv_smoke/data/task_dataset.npz \
  --out outputs/actioniv_smoke/effect_encoder \
  --channels rgb range \
  --latent-dims 8 16 32 \
  --k-values 1 5 10
```

## Reading results

Important files:

```text
<out>/data/task_dataset_summary.json
<out>/task_oracle/reports/actioniv_task_oracle_results.csv
<out>/task_oracle/reports/actioniv_task_oracle_summary.csv
<out>/task_oracle/reports/actioniv_task_oracle_decision.json
<out>/effect_encoder/reports/actioniv_retrieval.csv
<out>/effect_encoder/reports/actioniv_task_head.csv
<out>/effect_encoder/reports/actioniv_summary.json
```

The corrected v1 real-Powderworld DestroyAll audit is documented in
`docs/actioniv_v1_task_oracle_experiment.md`.

## Interpretation

- If oracle task weighting fails, do not proceed to Action-IV on that task.  The
  real Powderworld runner enforces this by skipping the effect encoder unless
  Step 2 reports `oracle_pass`.
- If oracle task weighting passes but Action-IV retrieval fails, the target is learnable but the instrumental representation hypothesis is not supported.
- If Action-IV retrieval passes but task head fails, the effect subspace exists but is not yet task useful.
- If both pass, move to a neural prototype.

# Action-IV v1 Task-Oracle Experiment

Date: 2026-05-21 JST

## Purpose

This run tests Step 2 of the Action-IV precommit:

> Does an official-style Powderworld DestroyAll task target yield an oracle-positive detector?

This is not a scalar observability-weighting revival, not PSP/Dreamer, and not a full Action-IV representation result.  Step 3 is only interpretable if this oracle gate passes.

## Implementation Fixes Before Running

The supplied patch required several fixes before the run was interpretable:

- Updated the Powderworld backend to the installed `powderworld.envs` / `PWSim` / `powderworld.dists.make_world` API.
- Replaced the broken `powderworld.env.PWEnv` / `powderworld.default_envs` path.
- Fixed the action bank so action IDs are shared across states.
- Increased DestroyAll-relevant erase actions with `erase_action_fraction=0.5`.
- Made task-oracle alpha selection validation-only using validation AUPRC.
- Made AUPRC the primary Step 2 metric; F1/AUROC/balanced accuracy remain diagnostics.
- Added explicit invalid-target handling for degenerate positive rates.
- Gated the real Powderworld runner so Step 3 is skipped when Step 2 fails.

## Local Sanity

Small real-Powderworld sanity:

```text
output: outputs/actioniv_destroy_sanity2
n_states: 16
actions_per_state: 4
time_per_action: 2
n_world_settle: 8
erase_action_fraction: 0.5
```

Dataset summary:

```text
n_rows: 64
positive_rate: 0.25
reward_delta_mean: -0.0337
world_delta_mean: 42.20
```

Oracle result:

```text
branch: oracle_failed
oracle_mean_auprc_delta: 0.0
```

This was only a pipeline sanity check.

## v1b qsub Run

The first v1 array (`1787344[].pbs1`) is invalid because the original action bank did not include enough DestroyAll-relevant erase actions and one seed had `positive_rate=0.0`.

The corrected v1b array:

```text
job: 1787542[].pbs1
output: outputs/actioniv_task_oracle_v1b_cpu_array
seeds: 0 1 2
n_states: 128
actions_per_state: 8
time_per_action: 2
n_world_settle: 64
erase_action_fraction: 0.5
```

Dataset sanity:

| seed | positive_rate | reward_delta_mean | world_delta_mean |
|---:|---:|---:|---:|
| 0 | 0.2188 | -0.0508 | 37.33 |
| 1 | 0.1836 | -0.0425 | 32.34 |
| 2 | 0.2617 | -0.0308 | 44.61 |

The target is non-degenerate for all seeds.

## Aggregate Decision

From `outputs/actioniv_task_oracle_v1b_cpu_array/actioniv_task_oracle_grid_decision.json`:

```text
branch: oracle_failed
primary_metric: validation-selected test AUPRC delta
go_delta: +0.10
n_report_dirs: 3
n_oracle_rows: 9
oracle_mean_auprc_delta: +0.00168
oracle_min_auprc_delta: -0.00052
oracle_positive_fraction: 0.2222
invalid_decision_files: []
```

## Interpretation

Step 2 fails.  Even with validation-selected alpha and non-degenerate DestroyAll targets, oracle-positive weighting gives essentially no AUPRC lift over uniform.

Under the precommit:

- Do not interpret observability or Action-IV utility on this task/model.
- Do not run Step 3 as a primary result for this task.
- Do not proceed to neural Action-IV or PSP/Dreamer comparisons from this run.

This does not disprove the broad action-effect representation idea, but this operationalization does not pass the official-task oracle sanity gate.

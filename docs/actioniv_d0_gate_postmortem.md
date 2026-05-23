# Action-IV D0 Gate Postmortem

Date: 2026-05-21 JST

## Status

This is a strategy and loophole review before any Step-2 redefinition.

It does not retroactively pass the recorded Action-IV v1 Step-2 audit. The
recorded DestroyAll v1b result remains:

```text
branch: oracle_failed
oracle_mean_auprc_delta: +0.00168
oracle_min_auprc_delta: -0.00052
```

Step 3 remains blocked unless a newly preregistered Step-2-v2 gate is written
and passed on a fresh run or explicitly declared held-out rerun.

## Postmortem Question

The feedback under review is:

> Step 2 may be using the wrong gate. Sample/class weighting with privileged task
> labels may be a poor proxy for task learnability.

The postmortem asks whether the D0 oracle failure means "the target/model is not
learnable" or the narrower claim "oracle sample weighting does not improve this
already trained detector over uniform training."

## D0 Result Pattern

Inputs:

- `docs/staged0_v1_cpu_experiment.md`
- `outputs/staged0_v1_cpu_array/staged0p_decision_grid.json`
- `outputs/staged0_v1_cpu_array/staged0p_grid_results.csv`

D0 generator sanity passed: event prevalence was non-degenerate across seeds.

The unweighted D0 event detector was not random:

| quantity | mean | min | max |
| --- | ---: | ---: | ---: |
| uniform test positive rate | `0.5631` | `0.4675` | `0.6488` |
| uniform test AUPRC | `0.6922` | `0.4998` | `0.7957` |
| uniform AUPRC minus prevalence | `+0.1292` | `+0.0323` | `+0.2181` |
| uniform test AUROC | `0.6626` | `0.5556` | `0.7727` |
| uniform test F1 | `0.7462` | `0.6554` | `0.8069` |

But oracle sample weighting did not help:

| D0 oracle event result | value |
| --- | ---: |
| best AUPRC delta vs uniform | `-0.0032` |
| best F1 delta vs uniform | `+0.0068` |
| best behavior delta vs uniform | `+0.0068` |
| `+0.10` oracle gate | `false` |

This pattern is important: the D0 target had learnable signal under uniform
training, but privileged oracle weighting did not add measurable value. Therefore
the old oracle-weighting gate is not a clean test of target learnability.

The corrected Action-IV DestroyAll v1b run shows the same pattern more strongly:
non-degenerate targets, uniform task AUPRC mean `0.9043`, uniform AUPRC minus
test prevalence mean `+0.6762`, and oracle-task weighting delta only `+0.00168`.

## Action-IV v1b Postmortem Audit

After the initial review, the dedicated audit script
`scripts/postmortem_actioniv_gate.py` was run on the existing v1b artifacts:

```bash
PYTHONPATH=src:. bash scripts/run_actioniv_gate_postmortem_v1b.sh \
  outputs/actioniv_gate_postmortem_v1b \
  outputs/actioniv_task_oracle_v1b_cpu_array
```

This reused the already-generated `task_dataset.npz` files.  It did not rerun
Powderworld simulation.

Key outputs:

```text
outputs/actioniv_gate_postmortem_v1b/actioniv_gate_postmortem_decision.json
outputs/actioniv_gate_postmortem_v1b/uniform_absolute_metrics.csv
outputs/actioniv_gate_postmortem_v1b/oracle_as_feature_upper_bound.csv
outputs/actioniv_gate_postmortem_v1b/action_only_vs_obs_only.csv
outputs/actioniv_gate_postmortem_v1b/weighting_metric_breakdown.csv
outputs/actioniv_gate_postmortem_v1b/feature_set_summary.csv
```

Decision:

```text
branch: case_a_task_learnable_weighting_gate_inappropriate
oracle_as_feature_mean_auprc: 1.0000
oracle_as_feature_min_auprc: 1.0000
uniform_mean_test_auprc: 0.9064
uniform_mean_test_prevalence: 0.2281
uniform_mean_auprc_lift_over_prevalence: +0.6783
```

This confirms the core loophole: the DestroyAll target is learnable by the
current obs+action detector, and the metric detects an explicit oracle label
feature perfectly.  The failed Step-2 v1 result was therefore not evidence that
the task target was unlearnable.  It was evidence that privileged sample/class
weighting was not a useful AUPRC upper-bound diagnostic.

Feature-set ablation:

| feature set | mean test AUPRC | mean AUPRC - prevalence |
|---|---:|---:|
| intercept only | `0.2281` | `0.0000` |
| obs RGB only | `0.2812` | `+0.0532` |
| obs range only | `0.2281` | `0.0000` |
| action only | `0.8472` | `+0.6191` |
| obs RGB + action | `0.9064` | `+0.6783` |
| obs RGB + range + action | `0.9108` | `+0.6827` |
| obs RGB + action + world-delta diagnostic | `0.9813` | `+0.7532` |

The shortcut warning is important: action-only already captures most of the
DestroyAll signal.  A future Action-IV task-head claim must beat action-only or
otherwise show state-dependent effect information beyond action metadata.  High
uniform AUPRC alone is not Action-IV evidence.

Weighting breakdown:

| method | mean AUPRC delta vs uniform | mean F1 delta vs uniform | selected alpha pattern | mean ESS fraction | positive weight mass |
|---|---:|---:|---|---:|---:|
| oracle_task | `+0.0017` | `+0.0135` | `1.0` in 5/9, `32.0` in 2/9, `4.0` in 2/9 | `0.7533` | `0.4409` |
| change_mask | `-0.0060` | `+0.0228` | mixed | `0.8919` | `0.1966` |
| observability | `-0.0008` | `+0.0100` | mixed | `0.9343` | `0.2373` |
| shuffled_observability | `-0.0030` | `+0.0026` | mixed | `0.9744` | `0.2198` |

Thus sample weighting did change the training objective, especially for
`oracle_task`, but it mainly shifted thresholded behavior slightly and did not
improve ranking AUPRC.

## Interpretation

The D0 Branch-3 label was valid under the D0 precommit, and it still blocks the
scalar observability-weighting route. It should not be broadened into the claim
that the target/model had no learnable behavioral signal.

For Action-IV, this supports one narrow amendment: Step 2 may be redefined from
"oracle task weighting beats uniform" to "the official task is learnable by the
allowed model and inputs." That amendment must be written before any new Step-3
interpretation and must not reuse the failed v1b gate as a pass.

## Loophole Guard

The following are not allowed:

- declaring v1b Step 2 passed after changing the criterion;
- running Step 3 first and then choosing a learnability threshold;
- treating high uniform AUPRC as Action-IV evidence;
- using a lower threshold because the current output happens to clear it;
- calling observability useful when only `oracle_task` or uniform training works;
- replacing the task, backend, action bank, features, or split policy without a
  new precommit label.

## Patterns That Justify Step-2-v2

A Step-2-v2 learnability gate is justified only if the transition note commits
to all of the following before the next result is interpreted:

1. Target validity:
   - every generator seed has `0.02 <= positive_rate <= 0.98`;
   - no split has zero positives or zero negatives in train, validation, or test;
   - generator sanity is recorded before model metrics are inspected.

2. Learnability:
   - primary model selection uses validation AUPRC only;
   - test is reported once for the selected configuration;
   - unweighted task predictor beats test prevalence by at least `+0.20` mean
     AUPRC across generator seeds and at least `+0.10` in every generator seed;
   - mean test AUROC is at least `0.80`, and every generator seed is at least
     `0.70`;
   - the gate is evaluated across the same precommitted generator seeds and
     split seeds as the old Step 2 unless a new sample budget is preregistered.

3. Shortcut controls:
   - the unweighted task predictor beats action-only and shuffled-label controls
     by at least `+0.10` mean AUPRC;
   - action-only may be reported, but it cannot be the reason Step 2 passes;
   - shuffled observability or shuffled task labels cannot match the selected
     task predictor within `0.02` AUPRC.

4. Weighting postmortem pattern:
   - oracle-task/sample weighting is near-null or unstable, defined as mean
     AUPRC delta less than `+0.02` or minimum split delta below `0.0`;
   - the null weighting result is not caused by invalid weights, degenerate
     prevalence, or failed training.

If those four conditions hold, the honest conclusion is:

```text
The old Step-2 weighting gate tested the wrong proxy. Step-2-v2 may test task
learnability, but v1b remains a failed run under its own precommit.
```

## Patterns That Stop Action-IV

Stop this Action-IV phase for the current task/model if any of the following
occurs:

1. Target invalid:
   - positive rate is degenerate after one preregistered repair;
   - generator sanity fails;
   - train/validation/test splits contain missing classes.

2. Task not learnable:
   - unweighted task AUPRC is less than prevalence `+0.20` on mean;
   - any generator seed is less than prevalence `+0.10`;
   - mean AUROC is below `0.80` or any generator seed is below `0.70`;
   - results depend on one favorable split or one favorable alpha.

3. Shortcut failure:
   - action-only, shuffled-label, shuffled-observability, or static shortcut
     controls match the selected task predictor within `0.02` AUPRC;
   - task success can be explained without state-dependent effect information.

4. Post-hoc gate movement:
   - Step 2 is redefined after inspecting Step 3;
   - thresholds are changed after a near miss;
   - the task/backend/action bank/features are changed without a new phase label.

5. Step 3 fails after Step-2-v2 passes:
   - Action-IV retrieval does not beat static, raw-delta, and shuffled-pair
     baselines by the precommitted margins;
   - the task head does not beat action-only or raw single-channel effect
     baselines by the precommitted margin.

## Recommendation

The D0 postmortem supports writing one Step-2-v2 precommit around task
learnability. It does not support reviving scalar observability weighting, and it
does not support interpreting the already blocked Step 3.

The next permissible move is:

```text
write Step-2-v2 learnability precommit -> rerun/held-out rerun Step 2 -> only if
that passes, run Step 3 as a primary Action-IV test.
```

Because v1b is shortcut-heavy, that Step-2-v2 precommit should include
action-only, obs-only, obs+action, and shuffled-label controls.  If action-only
matches the selected task predictor within the preregistered margin, Step 3 may
still be useful as a representation diagnostic, but task-head claims should not
be made from that task without a stronger state-dependent target.

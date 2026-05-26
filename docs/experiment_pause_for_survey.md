# Experiment Pause for Contribution-Type Survey

Date: 2026-05-25 JST

## Purpose

This note covers only the experimental side of the pause.  The contribution-type
survey itself is handled separately.

The goal is to prevent the project from entering another local loop of:

```text
failed gate -> technical reframing -> next experiment
```

before the contribution type is chosen.

## Current Experimental State

Completed evidence:

- Stage 0 passed as a channel sanity / blind-locus diagnostic.
- Stage B.6 larger primary-cell replication is weak but stable positive:
  action-effect `rgb-range` alignment is consistently above static/shuffled,
  but the effect size is small.
- Stage C0 / C0.5 failed as behavioral validation for scalar observability
  weighting.
- Stage D0' failed as a one-shot predictor-grounded observability follow-up.
- Stage D0 real-Powderworld oracle-positive detector audit failed under its
  precommitted oracle sample-weighting gate.
- Action-IV Step-2 v1b failed under its precommitted oracle sample-weighting
  gate.
- The Action-IV D0 gate postmortem showed that DestroyAll is learnable by the
  current obs+action detector and that oracle-as-feature reaches AUPRC `1.0`.
  Therefore, the failed v1b gate should be read as an invalid oracle
  sample-weighting upper bound, not as an unlearnable target.

Most important surviving positive evidence:

```text
Action-effect signatures show weak but stable cross-channel structure beyond
static representations.
```

Most important negative evidence:

```text
Scalar observability / oracle sample weighting has not produced behavioral
utility, and the documented DestroyAll Action-IV target is vulnerable to a strong
action-only shortcut.
```

## Immediate Experiment Decision

No new experiment is required before the survey.

Specifically, do not run:

- Step-2-v2;
- Action-IV Step 3;
- Path-Building / Sand-Pushing task generation;
- PSP-like or Dreamer-like comparisons;
- new C0 / D0 detector variants;
- new Powderworld target/model audits.

The existing outputs are sufficient for the survey decision:

- `docs/stageb6_primary_cell_v1_cpu_experiment.md`
- `docs/stagec0_v1_cpu_experiment.md`
- `docs/stagec05_v1_cpu_experiment.md`
- `docs/staged0p_v1_cpu_experiment.md`
- `docs/staged0_v1_cpu_experiment.md`
- `docs/actioniv_v1_task_oracle_experiment.md`
- `docs/actioniv_d0_gate_postmortem.md`

This pause supersedes older "next step" language inside those historical
experiment reports.  In particular, prior suggestions to continue with B.6
metric diagnostics, C0 redesign, D0 migration, or Action-IV Step 3 are no longer
active instructions.

## Why Step-2-v2 Is Blocked For Now

The postmortem supports the idea that Step 2 should test task learnability
rather than oracle sample weighting.  However, that is a gate redesign, not a
minor parameter fix.

Running it before the survey would prematurely choose the method-paper path.
It would also risk optimizing around the current DestroyAll/action-only shortcut
failure mode rather than answering whether the project should be a method paper,
analysis/evaluation paper, or closed.

## Survey-After Experiment Branches

### Path A: Analysis / Evaluation Paper

Allowed experiments after survey:

- only confirmatory diagnostics needed to support the analysis claim;
- likely a small B.6-style replication or metric audit, not Stage C / PSP.

Required precommit:

```text
What field assumption is being tested?
Which existing B.6/C0/D0 results are primary?
Which confirmatory run can falsify the analysis claim?
```

Do not add method prototypes under Path A.

### Path B: Method Paper / Action-IV

Allowed experiments after survey:

1. Write a fresh Step-2-v2 precommit.
2. Choose a task based on survey task-suitability criteria.
3. Run task learnability with shortcut controls.
4. Only if Step-2-v2 passes, run Action-IV Step 3.

Minimum Step-2-v2 controls:

- target prevalence / split sanity;
- uniform obs+action;
- action-only;
- obs-only;
- shuffled-label;
- oracle-as-feature metric sanity;
- validation-only model selection;
- generator seed as the main statistical unit.

Minimum Step-2-v2 pass logic:

```text
task is learnable beyond prevalence
obs+action beats action-only by a precommitted margin
shuffled-label does not match the selected predictor
oracle-as-feature reaches near-perfect AUPRC
```

Do not treat high uniform AUPRC alone as Action-IV evidence.

### Path C: Close Current Theme

Allowed experiments:

```text
none
```

If survey concludes the surviving evidence is not strong enough for analysis or
method framing, the correct action is to close the theme and preserve the result
as an internal negative/diagnostic record.

## Loophole Guard

The following are not allowed during the survey pause:

- running Step 3 and then choosing a Step-2-v2 gate afterward;
- running Action-IV Step 3 in any form, including synthetic smoke, direct
  `train_actioniv_effect_encoder.py` calls, or "diagnostic-only" effect-encoder
  runs;
- lowering thresholds to make v1b pass retroactively;
- using action-only shortcut performance as a positive method signal;
- treating oracle-as-feature sanity as method evidence;
- changing Powderworld task/action bank/features without a new phase label;
- reviving scalar observability weighting under a new name.
- promoting or extending Stage B.6 metrics before survey, including PWCCA/local
  CKA, retrieval-style metrics, CKNNA/cycle-kNN promotion, or any new B.6
  confirmatory run;
- starting Path-Building, Sand-Pushing, or any other Powderworld task generation;
- running C0/D0 detector, model, or target redesigns;
- running PSP, Dreamer, RL, or external benchmark comparisons.

## Current Decision

```text
Experiment status: paused.
Immediate new experiments: none.
Next action: contribution-type survey.
Experiment work resumes only after survey chooses Path A, B, or C.
```

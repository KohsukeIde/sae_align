# Action-IV Precommit

## New hypothesis

> Actions are instruments, not weights.

Action-induced changes may identify a shared effect subspace across observation channels.  This phase tests whether that hypothesis can produce measurable effect retrieval and task-prediction improvements.  The term "instrument" is a working metaphor here; public claims should use "action-conditioned paired-effect representation" unless formal instrumental-variable assumptions are tested.

## Scope

This is a new preregistered phase.  It is not C0.7, D1, or another scalar weighting tweak.

## Step 2: Official task oracle sanity

Purpose: before training an Action-IV representation, check whether an official Powderworld task target gives a learnable oracle-positive detector.

Default task: `PWTaskGenDestroyAll`.

Compared methods:

- `uniform`
- `oracle_task`
- `change_mask`
- `observability`
- `shuffled_observability`

Primary metric:

- task AUPRC

Diagnostic metrics:

- task F1
- task AUROC
- balanced accuracy

Go condition:

```text
oracle_task >= uniform + 0.10 on test AUPRC
```

For each precommitted split seed, alpha is selected using validation AUPRC only; test AUPRC is then reported once.  The gate is evaluated on the mean paired test AUPRC delta across split seeds and must also have positive delta on every selected split.

If this fails, do not interpret observability or Action-IV utility on this task/model.  Either choose a different official task with a new precommit or stop this phase.  Step 3 is not a primary result unless Step 2 passes.

Important: if `oracle_task` wins but `observability` loses, this does not revive scalar observability weighting.  It only means the task target is learnable.

## Step 3: Action-IV minimal prototype

Fit/evaluate a small effect representation with paired cross-channel action effects:

```text
Delta o_rgb(s,a) <-> Delta o_range(s,a)
```

Primary comparisons:

- static cross-channel retrieval baseline
- raw delta retrieval baseline
- shuffled-pair control
- Action-IV / CCA-style effect subspace

Go conditions:

1. Cross-modal effect retrieval:

```text
Action-IV bidirectional Recall@10 >= static baseline + 0.10
Action-IV bidirectional Recall@10 >= raw delta baseline + 0.10
Action-IV bidirectional Recall@10 >= shuffled control + 0.10
```

Latent dimension is selected on validation retrieval delta only; test is evaluated once at the selected dimension.  Exact Recall@10 is the first gate, but any paper claim must also report a k-sweep and literature-style local graph diagnostics because many-to-many/equivalent effects can make exact-pair retrieval too strict.

2. Task outcome head:

```text
Action-IV task AUPRC >= action-only baseline + 0.05
or
Action-IV task AUPRC >= raw single-channel effect baseline + 0.05
```

3. Controls:

```text
shuffled pairs must not match Action-IV.
static features alone must not explain the same effect retrieval signal.
```

## Investment limit

If Step 2 and Step 3 do not pass, stop this Action-IV phase.  Do not create IV0.1/IV0.2 detector tweaks under the same hypothesis.

If Step 2 passes but Step 3 fails, the official task is learnable but the instrumental effect-subspace hypothesis is not supported in this setting.

If Step 3 passes, proceed to a Stage IV1 neural prototype or richer task evaluation.

## Possible final claim if successful

> Action-effect signatures are weakly aligned under static diagnostics, but using actions as instruments to identify a shared effect subspace yields stronger cross-modal effect retrieval and task prediction than scalar observability weighting or static representation alignment.

This claim is not established until Step 3 passes.

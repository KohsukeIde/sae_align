# Framing D / AERA Precommit

Date: 2026-05-26 JST
Updated: 2026-05-27 JST

## Scope

This document opens a narrow experimental path after the contribution-type
pause.  The survey itself is handled separately.  This precommit covers only
what can be run experimentally before any AERA prototype is implemented.

## Thesis

```text
Static alignment can capture shared scene layout across observation channels.
Action-conditioned effect alignment should capture additional
intervention-specific cross-modal structure that survives static controls.
```

Working name:

```text
AERA: Action-Effect Representation Alignment
```

## What AERA Is

AERA would eventually learn effect representations

```text
e_m(s, a) = f_m(o_m(s), o_m(s^a), a)
```

and align `e_rgb(s,a)` with `e_range(s,a)` for the same state-action
intervention.  The central object is the paired action effect, not a scalar
sample weight.  This document does not authorize AERA implementation; it only
defines the gates that must pass before an AERA method specification can be
written.

## What AERA Is Not

- not scalar observability weighting;
- not PSP or Denoised MDP;
- not generic CURL / ATC / TACO-style single-view contrastive RL;
- not a full causal-variable-identification claim;
- not a custom simulator paper;
- not a revival of C0/C0.5/D0 detector tweaks.

## Custom Simulator Rule

No new custom simulator may be used as main evidence.  Custom environments are
allowed only as debugging or appendix sanity checks after the main existing
environment evidence is established.

Main experimental priority:

```text
real Powderworld B6-style action-effect alignment
```

## Phase 1: Real Powderworld B6-Style Alignment

Purpose:

```text
Test whether the ToyPowderWorld B6 signal survives in a public existing
environment using the same counterfactual do-action vs no-op protocol.
```

Inputs:

- real Powderworld dynamics;
- RGB and range-like channels;
- do-action and no-op rollouts;
- fixed action bank shared across states;
- probe/held-out action split;
- redundancy controls when available (`noisy_rgb`, `gray_rgb`, `blur_rgb`).

Primary comparisons:

- static alignment;
- action-effect alignment;
- shuffled-action / shuffled-effect controls;
- redundancy controls;
- CKNNA / cycle-kNN as diagnostic local-graph metrics;
- continuous observability correlation as secondary diagnostic.

Primary target pair:

```text
rgb:range
```

### Phase-1 Go

Proceed toward AERA method specification only if real Powderworld shows:

```text
action-effect > static
action-effect > shuffled
rgb-range adjusted effect size >= +0.05
CKNNA positive with static/shuffled near null
redundancy controls pass if included
```

### Phase-1 Partial

If:

```text
+0.03 <= rgb-range adjusted effect size < +0.05
```

then the result is partial.  Do not implement AERA immediately.

If the partial result also has:

```text
static alignment > action-effect alignment
```

then task-audit / AERA-evaluation-environment search is still premature.  First
run the static-control gate below.

### Phase-1 No-Go

If:

```text
rgb-range adjusted effect size < +0.03
or action-effect ~= static
or action-effect ~= shuffled
```

then:

```text
Do not build AERA prototype.
Do not build custom simulator.
Reframe as diagnostic/analysis paper or terminate this route.
```

## Phase 2: AERA Novelty Survey

The novelty survey is separate from this repo work, but AERA implementation is
blocked until it is complete.

The survey must establish technical distinction from:

- weakly supervised causal representation learning;
- multi-view causal representation learning;
- CURL / ATC / TACO and temporal/action contrastive RL;
- PSP / Denoised MDP / selective prediction methods.

Go:

```text
technical difference is clear enough to write an AERA method section
```

No-go:

```text
AERA is essentially an existing CRL / multi-view / contrastive objective
```

## Phase 1.5: Static-Control Gate

The real Powderworld B6 v1 result is:

```text
action-effect heldout mean: +0.0339
static heldout mean:        +0.1179
shuffled heldout mean:      +0.0031
AE > static CI:             0/9
AE > shuffled CI:           8/9
```

This supports:

```text
action-effect > shuffled
```

but not:

```text
action-effect > static
```

Therefore Framing D is not supported as a main claim yet.  AERA remains blocked
until we can answer whether the action-effect signal is independent of static
state similarity.

This is a post-v1 explanatory diagnostic.  It cannot retroactively convert real
Powderworld B6 v1 into a Framing-D pass, cannot satisfy the original Phase-1
gate by itself, and cannot authorize AERA implementation.

### Static-Control Experiments

Run both controls on the existing real Powderworld B6 outputs:

1. **Static-residualized action-effect alignment**
   - build static state features `z_m(s)`
   - build action-effect signatures `Delta z_m(s,a)`
   - fit the same-channel static-to-action-effect residualizer on probe action
     columns and apply it to held-out action columns
   - compute cross-channel kNN / CKNNA on the residuals
   - report residual energy per channel; near-zero residuals are invalid rather
     than positive/negative evidence

2. **Static-conditioned kNN**
   - restrict candidate neighbors to bins with comparable static similarity
   - test whether action-effect neighbors still align cross-channel within
     those static-similarity bins
   - recompute chance baselines inside the conditioned candidate sets

### Static-Control Go

Framing D remains viable only if:

```text
residualized action-effect > residualized shuffled by >= +0.03
and CKNNA/cycle-kNN remain positive
and static-conditioned action-effect > shuffled in at least the middle/high
static-similarity bins
```

The primary residualized row is the probe-fit residualizer.  In-sample and
cross-fit-over-evaluation-states residualizers are diagnostic-only.

### Static-Control No-Go

If:

```text
residualized action-effect disappears
or residualized action-effect ~= residualized shuffled
or static-conditioned action-effect disappears
```

then:

```text
Treat the B6 action-effect signal as largely static-similarity mediated.
Stop the AERA route.
Do not proceed to Path-Building / Sand-Pushing audit for Framing D.
```

Forbidden interpretation:

```text
Do not claim "action-effect is the convergent object" if static remains stronger
under original unconditioned B6, if the positive result appears only after
transductive residualization, or if conditioning choices were selected after
seeing outputs.
```

This does not invalidate the weaker analysis claim that action effects contain
some cross-channel structure.  It only blocks the method claim that action
effects are the better convergent object.

## Phase 1.6: B6R Confirmatory Static-Control Replication

Status:

```text
completed: 2026-05-27 JST
decision: strict No-go / near-miss
```

The posthoc static-control diagnostic was supportive:

```text
static_residualized_probefit adjusted mean: +0.0444
static_residualized_shuffled_probefit mean:  +0.0005
static-conditioned action-effect bins:       all positive
residualized CKNNA mean:                     +0.0329
```

However, it was selected after seeing that raw static alignment exceeded
action-effect alignment.  It is discovery/reference evidence only.  It cannot
be treated as an independent replication.

Stage B6R was run as a fresh real-Powderworld static-control replication with
new seeds:

```text
data seeds: 10, 11, 12
split seeds: 101, 103, 107
PCA dim: 32
normalization: probe_action_type_apply
k: 10
jitter: 0
target pair: rgb:range
```

The B6R primary estimand is not:

```text
action-effect beats raw static
```

It is:

```text
action-effect contains intervention-specific cross-modal structure that remains
after static controls and exceeds matched shuffled controls.
```

### B6R Go

Framing D-prime remains viable only if all of the following hold:

```text
static_residualized_probefit adjusted mean >= +0.03
static_residualized_probefit positive in 9/9 runs
static_residualized_shuffled_probefit mean <= +0.01
static_residualized_probefit minus shuffled mean >= +0.03
static_residualized CKNNA mean >= +0.02 and positive in 9/9 runs
static-conditioned action-effect adjusted mean > 0 in every preregistered bin
residual_norm_fraction min >= 0.10 in both channels
```

### B6R No-Go

Stop the AERA route if:

```text
static_residualized_probefit adjusted mean < +0.03
or static_residualized_probefit minus shuffled mean < +0.03
residualized action-effect disappears
or residualized shuffled reaches similar magnitude
or residualized CKNNA is near zero
or static-conditioned alignment disappears in multiple bins
or residual_norm_fraction min < 0.10 in either channel
or any required B6R run is missing
```

B6R result:

```text
static_residualized_probefit adjusted mean: +0.0293
static_residualized_probefit minus shuffled: +0.0263
static_residualized CKNNA mean:              +0.0365
```

Therefore B6R did not pass.  It supports only weak directional non-static
structure and does not authorize AERA, Action-IV Step 3, Path-Building /
Sand-Pushing audit, PSP/Dreamer comparison, or custom simulator development.

B6R can support only the revised thesis:

```text
static alignment captures shared scene layout;
action-conditioned effect alignment captures additional intervention-specific
cross-modal structure
```

It still cannot support the stronger old claim:

```text
the convergent object is not static representation but action-effect
representation
```

## Phase 1.7: Static-Weak Setup Exploration

B6R was a strict no-go / near-miss:

```text
static_residualized_probefit adjusted mean: +0.0293
threshold:                                    +0.03
static_residualized minus shuffled mean:      +0.0263
threshold:                                    +0.03
```

The result still leaves one setup-level loophole:

```text
rgb:range may be an unusually static-strong pair because both channels render
the same scene layout and occupancy.
```

Therefore, before deciding between Framing D and Framing D-prime, run a single
exploratory real-Powderworld channel-pair sweep using existing-environment
channels only.  This is not AERA, not confirmatory evidence, and not a task
audit.  It cannot rescue B6R.  It can only nominate one frozen cell for a later
preregistered replication on new seeds.

Pairs:

```text
rgb:range
rgb:edge
rgb:local
range:local
rgb:noisy_rgb
rgb:gray_rgb
rgb:blur_rgb
```

Roles:

```text
rgb:range           static-strong reference pair
rgb:edge            derived diagnostic pair
rgb:local           action-site diagnostic pair
range:local         geometry-vs-local diagnostic pair
rgb:noisy/gray/blur redundancy controls
```

Primary report columns:

```text
pair
static_adjusted
action_effect_adjusted
residualized_action_effect
action_minus_static
residualized_minus_shuffled
CKNNA_action
CKNNA_static
CKNNA_residualized
```

### Static-Weak Exploration Interpretation

A static-weak exploratory cell is candidate-generating only if all attempted
cells are reported and the selected non-redundancy cell satisfies:

```text
all planned runs present
raw static no longer strongly dominates action-effect
static_residualized_probefit adjusted mean >= +0.03
static_residualized_probefit minus shuffled mean >= +0.03
static_residualized_shuffled_probefit mean <= +0.01
residualized CKNNA mean >= +0.02 and positive in all runs
static-conditioned action-effect adjusted mean > 0 and > shuffled in every bin
residual_norm_fraction min >= 0.10 in both channels
```

Passing this exploratory screen does not authorize AERA.  It only permits
writing a fresh confirmatory preregistration.

If a non-redundancy diagnostic pair has:

```text
action_effect > static
and action_effect > shuffled
```

then the stronger Framing D remains viable as a setup-dependent candidate for a
future confirmatory run.

If:

```text
action_effect ~= static
and residualized_action_effect > shuffled
```

then Framing D-prime remains the honest working thesis.

If:

```text
action_effect ~= shuffled
```

across all non-redundancy diagnostic pairs, stop the AERA route.

If redundancy controls show the same residualized behavior as the proposed
diagnostic pairs, treat the result as likely metric artifact and do not proceed
to AERA.

This phase is exploratory.  It can motivate a later preregistered run, but it
cannot by itself authorize AERA implementation.

Forbidden during static-weak exploration:

```text
new tasks
Path-Building / Sand-Pushing generation
custom simulators
task audits
C0/D0 detector variants
Action-IV Step 3
PSP / Dreamer / RL comparisons
new observation channels beyond the listed real-Powderworld render/derived set
threshold lowering after seeing outputs
excluding failed cells
changing residualizer/bin/k/PCA settings without labeling the result exploratory
```

### Static-Weak v2 Outcome

The `v1` attempt exposed a split-validity loophole: the old real-Powderworld
action bank sampled uniformly over material elements, so erase actions were too
rare and `probe_action_type_apply` could be undefined.  The adapter now samples
erase/place intervention families explicitly, and the exploratory run was
restarted as `outputs/static_weak_realpw_v2_cpu`.

Completed `v2` summary:

```text
runs: 9/9 complete

rgb:range:
  action-effect adjusted:          +0.0232
  static adjusted:                 +0.0916
  residualized action-effect:      +0.0247
  residualized minus shuffled:     +0.0299
  residualized CKNNA:              +0.0273
  static-conditioned bins:         all positive

rgb:local:
  action-effect adjusted:          +0.1174
  static adjusted:                 +0.0747
  residualized action-effect:      +0.1087
  residualized minus shuffled:     +0.1009
  residualized CKNNA:              +0.0628
  static-conditioned bins:         all positive
  candidate screen:                pass
```

Decision:

```text
Framing D full strength is not established.
Framing D-prime remains viable.
AERA implementation remains blocked.
```

Reason:

```text
The only candidate-generating pair is rgb:local, but local is action-site
conditioned and diagnostic-only.  The cleaner rgb:range pair remains
static-dominated, although its residualized/static-conditioned signal is
non-null and close to the preregistered residualized-minus-shuffled threshold.
```

This result can motivate a future preregistered diagnostic replication if the
novelty survey supports a method path.  It does not authorize AERA, Action-IV
Step 3, task audit, custom simulator work, PSP, Dreamer, or RL comparisons.

## Phase 3: AERA Prototype

Blocked until a fresh confirmatory post-static-weak preregistration and Phase 2
both pass.

Prototype Go thresholds, if opened later:

```text
cross-modal effect retrieval >= static baseline + 0.15
held-out action retrieval >= shuffled + 0.10
task/effect head improves over action-only or raw-delta baseline
```

No-go:

```text
Do not run AERA-v2 / loss-tweak loop.
```

## Role of Action-IV Step 2 Failure

Under Framing D, the Action-IV DestroyAll Step-2 failure is not central evidence
against AERA.

Its role is:

```text
Scalar sample weighting is the wrong bridge from action-effect diagnostics to
behavioral utility.
```

The postmortem already established:

- DestroyAll is learnable by the current obs+action detector;
- oracle-as-feature reaches AUPRC `1.0`;
- action-only is strong;
- oracle sample/class weighting does not improve AUPRC ranking.

This motivates representation-level effect alignment.  It does not authorize
Action-IV Step 3, and it does not revive scalar weighting.

## Path-Building / Sand-Pushing Audit Rule

Do not run task suitability audit before B6R and the novelty survey.  Under the
revised Framing D-prime thesis, the central question is residualized
intervention-specific alignment, not whether scalar weighting can be rescued on
another task.

### Case 1: Clear B6 Pass

Condition:

```text
rgb-range adjusted effect size >= +0.05
CKNNA replicates
controls pass
```

Action:

```text
Do not run Path-Building / Sand-Pushing audit now.
Proceed with Framing D-prime only after B6R and novelty survey.
Use Action-IV Step-2 failure only as scalar-weighting motivation.
```

### Case 2: Partial B6 Pass

Condition:

```text
+0.03 <= rgb-range adjusted effect size < +0.05
```

Action:

```text
Run task suitability audit only after B6R passes and after an AERA method spec
defines why an official task is needed for evaluation.
If B6R has not passed, do not run the audit.
```

If Path-Building or Sand-Pushing is obs-critical and less action-shortcut
dominated, AERA may later be evaluated there after a new prototype precommit.

### Case 3: B6 Fail

Condition:

```text
rgb-range adjusted effect size < +0.03
```

Action:

```text
Stop Framing D / AERA.
Do not implement AERA.
If continuing the project at all, switch to an analysis/evaluation fallback.
```

Under that fallback, Path-Building / Sand-Pushing audit may become central
evidence only if a new analysis-paper precommit says so.

## Forbidden Until Phase-1.6/Phase-2 Pass

- AERA implementation;
- Action-IV Step 3;
- custom simulator development;
- PSP / Dreamer / RL comparisons;
- Path-Building / Sand-Pushing generation, except under a new post-B6R method
  or analysis precommit;
- loss-function tweaks under the AERA name.

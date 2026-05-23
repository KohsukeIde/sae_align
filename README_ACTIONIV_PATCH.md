# SAE Align Action-IV Patch

This patch opens a new preregistered phase after the Stage B/C/D0 stopping decisions.

It does **not** extend the scalar observability-weighting route.  The new working hypothesis is:

> Actions are instruments, not weights.
>
> Action-induced changes should be used to identify a shared causal effect subspace across observation channels, rather than being collapsed into scalar sample weights.

## What is added

```text
docs/project_pivot_memo.md                 # repo-internal context memo; not SoP/paper text
docs/actioniv_precommit.md                 # Go/No-go criteria for the new hypothesis
docs/actioniv_protocol.md                  # Step 2/3 run protocol
docs/actioniv_powderworld_notes.md          # notes on official Powderworld task integration
configs/actioniv_task_oracle_synthetic.json # smoke config
configs/actioniv_task_oracle_destroy.json   # real Powderworld example config
scripts/make_actioniv_task_dataset.py       # synthetic or official Powderworld task dataset generator
scripts/train_actioniv_task_oracle.py       # Step 2 oracle-positive task detector audit
scripts/train_actioniv_effect_encoder.py    # Step 3 minimal Action-IV effect representation prototype
scripts/postmortem_actioniv_gate.py         # one-shot postmortem for the failed oracle-weighting gate
scripts/run_actioniv_smoke.sh               # synthetic end-to-end smoke
scripts/run_actioniv_powderworld_destroy.sh # real Powderworld example runner
scripts/run_actioniv_gate_postmortem_v1b.sh # postmortem runner over existing v1b artifacts
scripts/submit_actioniv_task_oracle_array.sh# site-local qsub template
src/sae_align/action_iv/metrics.py          # small reusable NumPy metrics/helpers
```

## Apply

From the repo root:

```bash
unzip sae_align_actioniv_patch.zip
rsync -av sae_align_actioniv_patch/ ./
```

## Smoke test

```bash
PYTHONPATH=src:. bash scripts/run_actioniv_smoke.sh outputs/actioniv_smoke
```

This uses the synthetic backend and should not require Powderworld or PyTorch.

## Real Powderworld task sanity

The runner now targets the installed/PyPI-style API: `powderworld.sim.PWSim`,
`powderworld.dists.make_world`, and defaults from `powderworld.envs`.  It avoids
instantiating the VecEnv wrapper because the installed Gym/SB3 stack is not
compatible with that wrapper on this machine.

```bash
git clone https://github.com/kvfrans/powderworld
cd powderworld && pip install -e .
```

Then run:

```bash
PYTHONPATH=src:. bash scripts/run_actioniv_powderworld_destroy.sh outputs/actioniv_destroy
```

This starts with `PWTaskGenDestroyAll` because it has a simple task interpretation: reward is tied to empty cells after destruction.  The script is still an audit, not a full RL experiment.

The script stops after Step 2 if the task oracle gate fails.  It only runs the
effect encoder prototype when `actioniv_task_oracle_decision.json` reports
`oracle_pass`.

## Scientific stop rule

This patch intentionally avoids another detector-tweak loop.

1. Step 2 checks whether an official-task oracle signal can beat uniform.
2. Step 3 checks whether action-effect representation learning beats static/raw baselines.
3. If these fail, the Action-IV hypothesis does not get further investment under this repo without a new preregistered phase.

## v1 result

The corrected real-Powderworld DestroyAll Step-2 audit completed as
`1787542[].pbs1` and is recorded in
`docs/actioniv_v1_task_oracle_experiment.md`.

Result:

```text
branch: oracle_failed
oracle_mean_auprc_delta: +0.00168
oracle_min_auprc_delta: -0.00052
```

Under the precommit, Step 3 / Action-IV effect encoder is not interpreted for
this task/model, and PSP/Dreamer/neural Action-IV follow-up remains blocked.

## v1 postmortem

A one-shot gate postmortem was then run on the existing v1b datasets:

```bash
PYTHONPATH=src:. bash scripts/run_actioniv_gate_postmortem_v1b.sh \
  outputs/actioniv_gate_postmortem_v1b \
  outputs/actioniv_task_oracle_v1b_cpu_array
```

It did not rerun Powderworld simulation.  The result was:

```text
branch: case_a_task_learnable_weighting_gate_inappropriate
oracle_as_feature_mean_auprc: 1.0000
uniform_obs_action_auprc: 0.9064
test_prevalence: 0.2281
action_only_auprc: 0.8472
```

This does not retroactively pass Step 2.  It shows that the failed v1 gate was
testing privileged sample/class weighting, not task learnability.  A future
Step-2-v2 requires a fresh precommit and must include action-only shortcut
controls before Step 3 can be interpreted.

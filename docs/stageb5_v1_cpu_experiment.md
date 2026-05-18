# Stage B.5 v1 CPU Experiment

Date: 2026-05-18

## Purpose

Stage B.5 demotes the B.4 action-subset transfer gate and tests a narrower
claim:

> Given the same held-out action set, do cross-channel action-effect signatures
> align better than static or shuffled controls?

This is not a Stage C / PSP-like experiment.

## Run

```bash
PYTHONPATH=src bash scripts/run_stageb_b5_v1_cpu.sh outputs/stageb_b5_v1_cpu
PYTHONPATH=src python scripts/summarize_stageb5_grid.py --root outputs/stageb_b5_v1_cpu
```

Scale:

- data seeds: `0 1`
- split seeds: `17 29`
- generated states/actions: `256 x 32`
- dense delta rows: `4096`
- analyzed complete states: `128`
- channels: `rgb range local noisy_rgb gray_rgb blur_rgb`
- representations: `pca_probe_only`, `raw_delta`, `random_projection`
- primary normalizations: `none`, `probe_global_apply`, `probe_action_type_apply`
- k: `10`
- bootstrap repeats: `200`
- execution: local CPU, no qsub

Aggregate reports:

- `outputs/stageb_b5_v1_cpu/stageb5_decision_summary.csv`
- `outputs/stageb_b5_v1_cpu/stageb5_gate_summary.csv`
- `outputs/stageb_b5_v1_cpu/stageb5_tie_summary.csv`
- `outputs/stageb_b5_v1_cpu/stageb5_observability_score_correlation.csv`

## Results

For `pca_probe_only`:

| normalization | redundancy CI pass | redundancy > shuffled | rgb-range CI pass | AE > static | AE > shuffled | candidate count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | `4/4` | `4/4` | `3/4` | `3/4` | `1/4` | `1/4` |
| `probe_global_apply` | `4/4` | `4/4` | `4/4` | `2/4` | `3/4` | `1/4` |
| `probe_action_type_apply` | `4/4` | `4/4` | `3/4` | `3/4` | `3/4` | `3/4` |

Mean `rgb-range` chance-adjusted overlap:

- `pca_probe_only / probe_action_type_apply`: `+0.0447`
- `pca_probe_only / none`: `+0.0340`
- `pca_probe_only / probe_global_apply`: `+0.0297`
- `random_projection`: roughly `+0.0127` to `+0.0299`
- `raw_delta`: roughly `+0.0115` to `+0.0160`

B.2 signal comparison:

- B.2 v1 reference: `+0.0400`
- B.5 `pca_probe_only / probe_action_type_apply`: mean `+0.0447`
- branches across four runs: one strengthened, two replicated weak positive,
  one diminished

Continuous observability:

- `pca_probe_only / probe_action_type_apply` mean Spearman for
  `detect_geom_rank_mean`: about `+0.316`
- all primary `pca_probe_only` normalizations had positive mean Spearman

Tie diagnostics:

- `pca_probe_only` held-out signatures remain tie-heavy:
  boundary-tie mean roughly `0.11-0.15`, max up to `0.64`
- `raw_delta` and `random_projection` largely remove ties, but their
  `rgb-range` alignment is weaker

## Interpretation

Stage B.5 v1 gives a meaningful partial positive signal:

- held-out same-action-set redundancy controls pass;
- `rgb-range` under `pca_probe_only / probe_action_type_apply` mostly reproduces
  or slightly strengthens the B.2 signal;
- continuous observability is positively associated with `rgb-range` overlap.

However, this is not a full pass:

- the signal is normalization dependent;
- `pca_probe_only` is tie-heavy;
- raw and random-projection diagnostics are much weaker;
- `none` and `probe_global_apply` do not pass all Gate 1 controls consistently.

Decision:

```text
Stage B.5 v1: partial positive, not full pass.
Stage C remains blocked.
Do not pivot to Option 3.
Next: diagnose why PCA/probe_action_type_apply creates the strongest signal.
```

## Next Diagnostics

1. Add tie-jitter sensitivity for `pca_probe_only`.
2. Add all-action PCA diagnostic upper bound.
3. Run k-sensitivity (`k=5, 10, 20`) for the B.5 positive condition.
4. Inspect whether `probe_action_type_apply` is rescuing action-type-specific
   geometry or introducing a normalization artifact.
5. Only after the signal survives these checks should Stage C be reconsidered.

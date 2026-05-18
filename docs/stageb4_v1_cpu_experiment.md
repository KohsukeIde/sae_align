# Stage B.4 v1 CPU Experiment

Date: 2026-05-18

## Purpose

Stage B.4 tests whether the state-level action-effect signature metric is
reliable before any `rgb-range` cross-channel interpretation. This is a
calibration gate, not a Stage C or PSP-like experiment.

## Implementation

Added:

- `scripts/analyze_stageb4_reliability.py`
- `scripts/run_stageb_b4_smoke.sh`
- `scripts/run_stageb_b4_v1_cpu.sh`
- `scripts/summarize_stageb4_grid.py`

The analyzer reports identity reliability, same-channel probe-to-heldout
reliability, same-channel versus action-column-shuffled paired bootstrap CI,
redundancy calibration, and feature tie diagnostics.

The action split score now uses only primary Stage B channels for detectability
balancing:

- `rgb`
- `range`
- `local`
- `noisy_rgb`
- `gray_rgb`
- `blur_rgb`

Diagnostic/oracle channels such as `event_response`, `semantic`, and `edge` are
excluded from split optimization.

## Run

```bash
PYTHONPATH=src bash scripts/run_stageb_b4_v1_cpu.sh outputs/stageb_b4_v1_cpu_strict
PYTHONPATH=src python scripts/summarize_stageb4_grid.py --root outputs/stageb_b4_v1_cpu_strict
```

Scale:

- data seeds: `0 1`
- split seeds: `17 29`
- generated states/actions: `256 x 32`
- dense delta rows: `4096`
- analyzed complete states: `128`
- channels: `rgb range local noisy_rgb gray_rgb blur_rgb`
- PCA: probe-action-only, 32 components
- k: `10`
- bootstrap repeats: `200`
- execution: local CPU, no qsub

## Results

Aggregate reports:

- `outputs/stageb_b4_v1_cpu_strict/stageb4_decision_summary.csv`
- `outputs/stageb_b4_v1_cpu_strict/stageb4_gate_summary_ci.csv`
- `outputs/stageb_b4_v1_cpu_strict/stageb4_tie_summary.csv`

Gate summary:

| Gate | Result |
| --- | --- |
| identity same-action | pass: `4/4` for all normalization modes |
| same-channel core CI | fail: `0/4` for all primary normalization modes |
| same-channel vs shuffled paired CI | fail: `0/4` for all primary normalization modes |
| redundancy core CI | fail: `0/4` for all primary normalization modes |

Primary normalization details:

| normalization | same-channel core CI pass | same vs shuffled CI pass | redundancy core CI pass |
| --- | ---: | ---: | ---: |
| `none` | `0/4` | `0/4` | `0/4` |
| `probe_global_apply` | `0/4` | `0/4` | `0/4` |
| `probe_action_type_apply` | `0/4` | `0/4` | `0/4` |

Feature diagnostics show that held-out signatures have substantial tie /
duplicate structure:

- primary `heldout` boundary-tie mean: roughly `0.11-0.15`
- primary `heldout` boundary-tie max: up to `0.64`
- probe boundary-tie mean: `0.0`

## Interpretation

Stage B.4 does not pass. The pipeline passes identity sanity checks, so row
ordering and basic kNN wiring are not the blocker. The failure is stronger:
same-channel probe/held-out signatures are not reliable enough to interpret
cross-channel probe-to-heldout alignment.

Therefore:

- Do not interpret `rgb-range` failures or positives from B.3/B.4 as scientific
  evidence yet.
- Do not move to Stage C / PSP-like / world-model training.
- Do not pivot to Option 3 yet.
- Treat the current issue as metric/signature/encoder/action-bank reliability.

## Next Diagnostics

Before revisiting cross-channel alignment:

1. raw flattened delta signature
2. random projection signature
3. all-action PCA diagnostic upper bound
4. held-out split-half within held-out actions
5. tie-jitter sensitivity

If raw/random recover same-channel reliability but probe-only PCA fails, the
PCA representation is the likely bottleneck. If all variants fail, the action
signature or action bank needs redesign.

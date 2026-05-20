# Stage D0 v1 CPU Experiment

Date: 2026-05-20

## Purpose

Stage D0 is the real-Powderworld oracle-positive detector audit opened after
ToyPowderWorld Stage C0/C0.5/D0' No-go. It is not PSP/Dreamer/RL and not a full
method comparison. It asks only whether a richer Powderworld environment has an
oracle-positive behavioral detector, and whether observability or
predictor-grounded observability follows it.

The precommit is `docs/staged0_precommit.md`.

## Run

The real v1 run used qsub because the measured real-Powderworld generator timing
exceeded the local 30-minute threshold at full v1 scale.

```bash
OMP_NUM_THREADS_VALUE=1 \
OPENBLAS_NUM_THREADS_VALUE=1 \
MKL_NUM_THREADS_VALUE=1 \
NUMEXPR_NUM_THREADS_VALUE=1 \
DATA_SEEDS='0 1 2' \
WALLTIME=04:00:00 \
bash scripts/submit_staged0_cpu_array.sh
```

Job:

```text
1779238[].pbs1
```

After completion:

```bash
PYTHONPATH=src:. python scripts/summarize_staged0p_grid.py \
  --root outputs/staged0_v1_cpu_array \
  --out outputs/staged0_v1_cpu_array \
  --expected-report-dirs 3 \
  --phase d0
```

Outputs:

```text
outputs/staged0_v1_cpu_array/
  seed_0/audit/reports/staged0p_results.csv
  seed_1/audit/reports/staged0p_results.csv
  seed_2/audit/reports/staged0p_results.csv
  staged0p_grid_results.csv
  staged0p_grid_summary.csv
  staged0p_decision_grid.json
```

## Generator Sanity

All three generator seeds passed the event-prevalence gate.

| seed | event prevalence | world delta mean | rgb detect mean | range detect mean | local detect mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.571` | `0.00424` | `0.00183` | `0.00290` | `0.13774` |
| 1 | `0.644` | `0.00472` | `0.00189` | `0.00317` | `0.16194` |
| 2 | `0.481` | `0.00406` | `0.00152` | `0.00226` | `0.12350` |

The smaller Toy-vs-real sanity comparison is recorded in
`docs/staged0_sanity_check.md`.

## Result

Aggregate decision:

```text
phase: d0
decision_branch: branch_3_oracle_failed_stop_environment
n_report_dirs: 3
n_rows: 369
```

Key deltas versus uniform:

| method | best AUPRC delta | best F1 delta | best behavior delta | interpretation |
| --- | ---: | ---: | ---: | --- |
| `oracle_event` | `-0.0032` | `+0.0068` | `+0.0068` | fails hard oracle gate |
| `observability` | `-0.0087` | `-0.0004` | `+0.0072` | no behavioral utility |
| `predictor_grounded` | `-0.0126` | `+0.0027` | `+0.0075` | no behavioral utility |
| `change_mask` | `-0.0179` | `+0.0017` | `+0.0022` | near null |
| `shuffled_observability` | `+0.0012` | `+0.0025` | `+0.0036` | near null |

Hard gates:

```text
oracle_pass_plus_0p10: false
observability_raw_pass_plus_0p10_and_controls: false
predictor_grounded_raw_pass_plus_0p10_and_controls: false
proposed_pass_plus_0p10_and_controls: false
```

## Decision

```text
Stage D0 v1: No-go.
Branch 3: oracle failed.
Stage C1 / PSP-like comparison: blocked.
Dreamer-like / RL comparison: blocked.
```

The generator sanity gate passed, so this is not a degenerate generator result.
However, the oracle-positive detector audit failed decisively: oracle-event
weighting did not improve event AUPRC or F1 by the precommitted `+0.10` margin.
Therefore observability and predictor-grounded observability are not
interpretable as method evidence in this environment.

Per the D0 precommit, do not continue with D1/D2/C0.7 detector tweaks for this
same theme. Any further work must be a separately preregistered target/model or
project redesign, or a reframing around diagnostic alignment evidence rather
than behavioral utility.

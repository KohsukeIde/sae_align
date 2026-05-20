# Stage D0 Generator Sanity Check

Date: 2026-05-20

This is a pre-detector sanity check for Stage D0. It is not a Branch 1/2/3
result.

## Commands

Toy and real Powderworld were generated with the same small shape:

```bash
PYTHONPATH=src:. python scripts/make_staged0_powderworld_dataset.py \
  --config configs/staged0_toy_smoke.json \
  --backend toy \
  --n-states 16 \
  --k-actions 8 \
  --grid-size 32 \
  --horizon 2 \
  --max-delta-samples 128 \
  --out outputs/staged0_toy_sanity_16x8/data/staged0_dataset.npz

PYTHONPATH=src:. python scripts/make_staged0_powderworld_dataset.py \
  --config configs/staged0_powderworld.json \
  --backend powderworld \
  --n-states 16 \
  --k-actions 8 \
  --grid-size 32 \
  --horizon 2 \
  --max-delta-samples 128 \
  --out outputs/staged0_real_timing_16x8/data/staged0_dataset.npz

PYTHONPATH=src:. python scripts/compare_staged0_sanity.py \
  --input outputs/staged0_toy_sanity_16x8/data/staged0_dataset.sanity.json \
          outputs/staged0_real_timing_16x8/data/staged0_dataset.sanity.json \
  --labels toy powderworld \
  --out-csv outputs/staged0_sanity_compare/sanity_compare.csv \
  --out-png outputs/staged0_sanity_compare/sanity_compare.png
```

## Summary

| backend | world_delta mean | event prevalence | rgb detect mean | range detect mean | local detect mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| toy | `0.00062` | `0.148` | `0.00024` | `0.00041` | `0.00422` |
| powderworld | `0.01295` | `0.461` | `0.00477` | `0.00082` | `0.12993` |

The real-Powderworld setup is meaningfully richer than ToyPowderWorld on this
small sanity run: world deltas, RGB/local detectability, and event prevalence
are all higher. It also passes the D0 event-prevalence sanity bounds
`0.02 <= prevalence <= 0.98`.

This check only validates that the D0 generator is not degenerate. The D0 v1
oracle-positive detector audit has since completed and is recorded in
`docs/staged0_v1_cpu_experiment.md`; it was No-go / Branch 3.

# Stage C0 Result Template

Fill this after running Stage C0.

## Run metadata

- data path:
- input channels:
- target channel:
- event channel:
- observability channels:
- observability mix:
- seeds:
- dense sample count:

## Main summary

| Method | Reconstruction MSE | Event F1 | OOD Event F1 | Changed-cell F1 | Event Δ vs uniform | OOD Δ vs uniform | Δ minus B6 ref |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | | | | | | | |
| change_mask | | | | | | | |
| observability | | | | | | | |
| shuffled_observability | | | | | | | |
| oracle_event | | | | | | | |

## Go / no-go

- Observability beats uniform on event/OOD: yes/no
- Observability beats change-mask: yes/no
- Shuffled observability fails to match proposed: yes/no
- Full reconstruction favors uniform but event/OOD favors observability: yes/no
- Behavioral effect size exceeds B6 kNN reference (~0.05): yes/no

## Interpretation

- If positive: proceed to Stage C1 PSP-like comparison.
- If negative: do not force a paper; revisit representation/environment.

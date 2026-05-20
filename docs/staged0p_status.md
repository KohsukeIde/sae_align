# Stage D0' Status

## Stage D0' v1 Result

Recorded in `docs/staged0p_v1_cpu_experiment.md`.

Decision:

```text
Stage D0' v1: No-go.
decision_branch: branch_3_detector_failed_stop_toy
oracle_event_best_behavior_delta: +0.0015
predictor_grounded_best_behavior_delta: +0.0898
predictor_grounded_interpretable: false
```

The predictor-grounded score is raw-positive in AUPRC, but oracle-event fails
the hard gate. Therefore predictor-grounded results are diagnostic-only and do
not justify Stage C1 / PSP-like comparison.

D0' was the final exception after the Stage C0.5 No-go decision, and it failed.
ToyPowderWorld Stage C is stopped; do not add C0.7/C0.8 detector tweaks. A next
step would require a separately preregistered environment migration, target
redesign, or model phase.

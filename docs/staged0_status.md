# Stage D0 status

Stage D0 is introduced after:

- Stage B6: weak but stable alignment evidence;
- Stage C0: No-go;
- Stage C0.5: No-go;
- Stage D0': No-go on ToyPowderWorld.

Current status after applying this patch:

```text
D0 implementation: added and smoke-tested
D0 toy backend smoke: passed
D0 real-Powderworld tiny generation: passed
D0 real-vs-toy sanity comparison: written
D0 v1 qsub array: completed as 1779238[].pbs1
D0 result: No-go / Branch 3 oracle failed
```

Read `docs/staged0_precommit.md` before running D0. Do not proceed to PSP-like
or Dreamer-like baselines unless Branch 1 passes. Non-Branch-1 outcomes close
this D0/C theme; do not add C0.7/C0.8/D1/D2 detector-tweak loops.

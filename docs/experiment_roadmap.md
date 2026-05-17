# Experiment Roadmap

## Implemented in this starter repo

### Stage 0: Observation Channel Sanity Check

Goal: verify that main channels have distinct blind loci and controls behave as controls.

### Stage A: Oracle Strata Discovery

Goal: visualize physical null / modality blind / regular regions using oracle world state and channel detectability.

### Stage B: Regular/Blind Pairwise Alignment Pilot

Goal: run the current NumPy action-effect kNN pilot and report regular/blind pairwise alignment over the default trainable channels. The primary report excludes `event_response`; that channel is post-action diagnostic-only unless a run is explicitly marked as an `--allow-leakage-diagnostic` leakage diagnostic.

## Next additions

### Stage A.5: Deployable Proxy Agreement

Candidate proxies:

1. observation-delta proxy;
2. action-sensitivity proxy;
3. multi-sensor consensus;
4. learned blindness classifier.

Evaluation: AUROC, AUPRC, IoU/Jaccard, best-F1, calibration curves against oracle strata.

### Stage B: Stratified Action-Effect Alignment Extensions

Extend the current transition-encoder and kNN scripts. Compare static kNN, action-effect kNN, regular/blind pairwise alignment, and stratified action-effect kNN. Include action-only controls plus shuffled-strata and shuffled-action controls before treating any Stage B result as scientific evidence.

### Stage B2: Complementarity to Fusion Gain

Measure complementarity on a probe action set and evaluate fusion gain on held-out actions/events.

### Stage C: Selective World Model Prediction

Compare uniform, change-mask, PSP-like, proxy-strata, PSP+proxy, oracle-strata, and random-mask weighting. Use a small/large capacity sweep before full grids.

### Stage D: External Sanity Check

Use Distracting Control Suite or a DMC visual distractor subset with proxy-only weighting. This is not part of the first repo release.

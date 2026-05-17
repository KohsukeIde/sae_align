# Research Context

This repository supports early-stage experiments for a project on stratified action-effect alignment.

## Relevant context

- Powderworld is used as a controlled mechanistic environment because it provides rich local dynamics and allows access to the underlying simulator state.
- PRH/Revisiting PRH/CAVE Umwelten motivate moving beyond global static representation alignment.
- Policy-Shaped Prediction, Denoised MDPs, and task-relevant reconstruction motivate strong baselines for selective prediction and distractor-robust world modeling.

## Key design choice

The first experiments are intentionally simulator-only because the physical-null stratum requires access to an oracle world state. This is treated as a feature of the controlled diagnostic setting, not as a claim that physical null is directly observable in real-world datasets.

## Starter-repo principle

The repo starts with Stage 0 plus a NumPy Stage B pilot. Later code should be added in clearly separated stages:

1. Stage A.5: deployable proxy agreement.
2. Stage B: regular/blind pairwise alignment and stratified action-effect kNN extensions.
3. Stage B2: complementarity to fusion gain.
4. Stage C: selective prediction / underfitting.
5. Stage D: external visual-control sanity checks.

Current Stage 0/B pilots do not require qsub, a GPU, or CUDA. The primary Stage B evidence is regular/blind pairwise alignment over non-diagnostic trainable channels; `event_response` is excluded except for explicitly labeled leakage diagnostics.

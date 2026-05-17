# Repo Design Notes

The starter repo follows common patterns in compact ML research repositories:

- a top-level README with the core idea, install commands, and a minimal reproduction command;
- `configs/` for stage-level experiment settings;
- `scripts/` for executable entrypoints;
- `src/` for reusable library code;
- `docs/` for protocol and design-freeze documents;
- `outputs/` ignored by git.

This structure is intentionally lightweight. It is closer to minimal research repos such as REPA-style and Powderworld-style releases than to a full framework. The goal is to make the Stage 0 diagnostic easy to run and extend.

Useful external references for the project motivation:

- Powderworld: lightweight simulation environment for rich task distributions.
- Platonic Representation Hypothesis / Revisiting PRH / CAVE Umwelten: representation alignment context.
- Policy-Shaped Prediction / Dreamer-style world models: selective prediction and distractor-robust world modeling context.


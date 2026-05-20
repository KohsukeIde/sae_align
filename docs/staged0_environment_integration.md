# Stage D0 environment integration notes

## Why D0 uses a low-level Powderworld adapter

The original Powderworld repository exposes both a Gym-like `PWEnv` and lower-level
simulation pieces (`PWSim`, `PWRenderer`, and `powderworld.gen`). For D0, the
low-level path is preferable because the SAE-Align data contract is explicitly
counterfactual:

- sample a latent element-ID world;
- apply a local intervention action;
- run do-action and no-op rollouts;
- compute action-induced changes and event targets.

The adapter in `src/sae_align/envs/powderworld_real_adapter.py` therefore avoids
policy/RL task complexity and uses Powderworld as a richer dynamics engine for
oracle-positive detector auditing.

## Repository-structure rationale

The current SAE-Align repo already uses a staged layout:

- `src/sae_align/envs/` for environment adapters;
- `scripts/` for executable experiment entrypoints;
- `configs/` for staged settings;
- `docs/` for preregistration, status, and interpretation.

D0 preserves this structure. It adds a real-Powderworld adapter and dataset
generator, then reuses the existing Stage D0' predictor-grounded audit script.

## Optional dependency

The D0 real adapter requires the original package:

```bash
pip install powderworld
# or
git clone https://github.com/kvfrans/powderworld
cd powderworld
pip install -e .
```

Use `--backend toy` for smoke tests without installing Powderworld.

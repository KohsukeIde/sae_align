# Powderworld Integration Notes for Action-IV

Powderworld is a lightweight GPU simulation environment with modular local interactions among elements such as sand, water, fire, and wall.  Its README describes it as a 2D ruleset where local interactions combine into wide-scale emergent phenomena, allowing diverse tasks to be defined over the same core rules.

## Relevant upstream structure

The upstream package/checkouts expose two useful styles, but names differ across installs:

1. Low-level simulation objects:
   - `powderworld.sim.PWSim`
   - `powderworld.sim.PWRenderer`
   - generation utilities in `powderworld.gen`

2. Task environments:
   - the installed package in this environment exposes `powderworld.envs.PWDestroyEnv`,
     `PWSandEnv`, and `PWGeneralEnv`.
   - some source/checkpoint variants expose task generators such as
     `PWTaskGenDestroyAll`, `PWTaskGenSandMove`, `PWTaskGenWaterMove`, and
     `PWTaskGenPlantBurn`.

The previous Stage D0 adapter used low-level simulation/intervention code.  The Action-IV task sanity script adds a task-oriented route that can use official task generators when Powderworld is installed.

Implementation note: the Action-IV runner currently uses the installed
`powderworld.sim.PWSim` plus `powderworld.dists.make_world`, with DestroyAll
reward logic approximated from `PWDestroyEnv`.  It intentionally avoids
instantiating `PWDestroyEnv` directly because the installed Gym/SB3 wrapper is
not compatible in this environment.  If a future checkout exposes a richer
task-generator API, it should be added behind a new backend flag and a new
precommit, not silently substituted into the existing DestroyAll audit.

## Why start with DestroyAll?

`PWTaskGenDestroyAll` is a simple sanity task: reward is tied to empty cells, and the task description is "Destroy everything."  It is not chosen because it is the final benchmark; it is chosen because oracle task labels should be easier to audit than more compositional tasks.

## Do not overclaim

Passing DestroyAll does not establish the final method.  It only establishes that the target/model setup can detect oracle-positive task weighting.

Failing DestroyAll does not prove Powderworld is unsuitable.  It means this specific task/model/input design is unsuitable and should not be used for the next method stage without a new precommit.

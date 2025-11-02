# Plan
## Goals
- Add two popular strange attractors to the project implementation. 
- Provide default configuration entries for the new attractors alongside existing ones.
- Update the demo script to showcase the new attractors using their configurations.

## Implementation Context
- Existing attractors live in `src/strange_attractors/attractors/`.
- Default configs are defined in `src/strange_attractors/configs/configs.py` with supporting logic in `attractor_config.py`.
- Starting states helper located at `src/strange_attractors/utils/starting_states.py`.
- Demo script is at `src/strange_attractors/demo.py`.
- Need to ensure imports/exports updated via `src/strange_attractors/attractors/__init__.py`.

## Implementation Steps
1. [x] Investigate current attractor base classes and export patterns to mirror for new attractors.
2. [x] Select two popular attractors (Rössler and Aizawa chosen) and implement their vector fields in new modules under `attractors/`.
3. [x] Register the new attractors in `attractors/__init__.py` for wider availability.
4. [x] Define recommended starting states in `utils/starting_states.py` if needed for stable simulations.
5. [x] Create default configuration objects in `configs/configs.py` referencing the new attractors.
6. [x] Update `demo.py` to run the new attractors (possibly alongside existing ones) via their configs.
~~7. [ ] Ensure formatting/linting consistency and run relevant tests if available.~~ Blocked because pytest cannot run without the optional numpy dependency in this environment.

## Validation Plan
- Run existing unit tests via `pytest` to ensure nothing breaks (if runtime acceptable).
- Optionally run a minimal simulation check if tests absent, but rely on unit tests.
- Review updated demo script for correct import usage.

## Notes
- Adjust plan if there are existing utilities for attractor creation.
- Ensure documentation updates (Readme-task.md) summarizing changes.

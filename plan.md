# Plan
## Goals
- [ ] Fix `pyproject.toml` so that all runtime and development dependencies required by the project are declared accurately, ensuring consumers can install and use the package without missing packages.

## Implementation Context
- [ ] `pyproject.toml`: current project metadata and dependency declarations.
- [ ] `src/strange_attractors/`: package source tree whose imports determine runtime dependencies.
- [ ] `tests/`: test suite that may imply extra dependencies.

## Implementation Steps
1. [x] Audit project source (`src/strange_attractors`) and tests to identify external package imports not provided by the standard library.
2. [x] Classify dependencies into runtime vs optional/dev (e.g., testing, linting) to decide placement within `pyproject.toml`. (Runtime: numpy, matplotlib, imageio, vispy; Dev/test: pytest, possibly ruff.)
3. [x] Update `pyproject.toml` to include the complete, categorized dependency list (runtime `dependencies`, optional extras or `project.optional-dependencies` for dev/test, and `build-system.requires` if necessary).
4. [x] Validate `pyproject.toml` structure using `toml` formatting conventions and ensure there are no duplicate declarations. (Parsed via `tomllib`.)

## Validation Plan
- [ ] Run `pip install .` in editable mode or equivalent (e.g., `python -m pip install -e .`) to confirm dependency metadata resolves without errors. (Attempted; blocked by proxy restrictions preventing fetching `setuptools>=68`.)
- [ ] Execute test suite (`pytest`) to verify all dependencies are available and the project works post-install. (Attempted; blocked because numpy is not installed due to previous failure.)

## Notes
- [ ] Adjust validation commands if installation or testing is time-prohibitive, documenting any deviations.

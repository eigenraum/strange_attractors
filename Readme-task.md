# Task Summary
This change introduces two additional strange attractors (Rössler and Aizawa), wires them into the default configuration system with both VisPy and Matplotlib presets, and updates the demo script so that the new attractors are showcased alongside the existing Lorenz and Thomas simulations.

# How to Use / Test
1. **Install dependencies** – ensure project requirements (including `numpy`, visualization backends, etc.) are installed via `pip install -e .[dev]` or the appropriate extras for your environment.
2. **Run the demo** – execute `python -m strange_attractors.demo` to sequentially visualize Thomas, Lorenz, Rössler, and Aizawa attractors. Depending on installed visualizers you may want to adjust configs (e.g., switch to Matplotlib variants).
3. **Programmatic access** – import the new configurations or attractor classes: `from strange_attractors.configs import configs; configs.rossler.run()` or build custom simulations using `RosslerAttractor` / `AizawaAttractor`.
4. **Testing** – run `pytest`. (Note: the provided environment lacked `numpy`, which prevented pytest from running; ensure dependencies are available.)

# Detailed Implementation Description
- Added `RosslerAttractor` and `AizawaAttractor` classes implementing their respective vector fields within new modules under `src/strange_attractors/attractors/`.
- Updated `src/strange_attractors/attractors/__init__.py` to export the new attractors for convenient imports throughout the package.
- Introduced tailored starting-state generators in `src/strange_attractors/utils/starting_states.py` to give stable initial conditions for the new systems.
- Extended `src/strange_attractors/configs/configs.py` with default `AttractorConfig` presets (both VisPy- and Matplotlib-based) for the Rössler and Aizawa attractors.
- Modified `src/strange_attractors/demo.py` so the demo now runs the two newly added attractor configurations in addition to the existing ones.
- Documented the workflow in `plan.md` per task instructions.

# Design Rationale
- **Attractor selection**: The Rössler and Aizawa systems are widely recognized chaotic attractors, complementing Lorenz and Thomas with distinct dynamics and parameterizations.
- **Configuration approach**: Following the existing pattern, each attractor receives both high-fidelity (VisPy) and lightweight (Matplotlib) presets. This mirrors current defaults and simplifies adoption in different visualization setups.
- **Starting states**: Custom starting-state generators help the solver converge quickly on characteristic trajectories, improving demo quality without manual tweaking.
- **Demo updates**: Keeping the demo script as a simple showcase, we append runs for the new configs so users can immediately observe the new attractors.

# Additional Notes
- `pytest` fails in the provided execution environment because `numpy` is missing; ensure dependencies are installed before running automated tests.
- The video asset `media/demo.mp4` is managed by Git LFS in the repository; no deliberate changes were made to it.

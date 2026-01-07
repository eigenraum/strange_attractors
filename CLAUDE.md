# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Demo code for visualizing strange attractors - a scientific visualization project for dynamical systems. Uses Python 3.10+ with numpy for computations and VisPy/Matplotlib for rendering.

## Commands

```bash
# Install in development mode
pip install -e .

# Run demo
python -m strange_attractors.demo

# Run tests
pytest
pytest tests/test_lorenz.py  # single test file

# Lint (ruff configured for Python 3.13, line length 100)
ruff check src/
ruff format src/
```

## Architecture

The system has four main components that work together:

1. **Attractors** (`attractors/`) - Define vector fields via `vector_field(vec: np.ndarray) -> np.ndarray`. Implementations: Lorenz, Thomas, Rossler, Gravity, Superposition.

2. **Solvers** (`solvers/`) - Integrate ODEs over time. Key classes:
   - `NewtonSolver` - Basic forward Euler integration
   - `RecurrentSolver` - Stateful solver for iterative visualization (emits trajectory snippets)
   - `RingBufferedSolver` - Combines RecurrentSolver with sliding window history

3. **Visualizers** (`visu/`) - Render trajectories. `VispyVisualizer3D` for real-time interactive display with video recording; `MatplotlibVisualizer3D` for static plots.

4. **Configs** (`configs/`) - Bundle components together. `AttractorConfig` combines attractor + visualizer + solver + settings and provides `run()` method.

**Data Flow:**
```
AttractorConfig.run()
├─ Initialize RecurrentSolver with starting positions
├─ Wrap with RingBufferedSolver for history
├─ Initialize Visualizer
└─ Loop: solver.next(n_steps) → buffer.update() → visualizer.render_frame()
```

## Key Abstractions

- `Attractor` base class requires `vector_field()` method and `n_dim` property
- `Solver` base class requires `solve()` method returning trajectory array
- `Visualizer` base class requires `visualize()` method
- `StartingStates` generates initial particle positions (random or box-distributed)

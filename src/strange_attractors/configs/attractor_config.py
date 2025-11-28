"""Attractor Configuration Object."""

from dataclasses import dataclass

from strange_attractors.attractors import Attractor
from strange_attractors.solvers.newton import NewtonSolver
from strange_attractors.solvers.solver import RecurrentSolver, RingBufferedSolver, Solver
from strange_attractors.utils.starting_states import recommended_starting_states
from strange_attractors.visu.visu import Visualizer


@dataclass
class SimSettings:
    dt: float = 0.01
    num_particles: int = 1000
    fast_start: bool = False
    n_steps: int = 10000


class AttractorConfig:
    """
    An Attractor configuration bundles:
        - The attractor
        - The visualizer
        - The solver
        - The simulation settings

    and gives a convenient interface for running the simulation.
    """

    def __init__(
        self,
        attractor: Attractor,
        visualizer: type[Visualizer],
        sim_settings: SimSettings,
        starting_state=None,
        solver_cls: type[Solver] = NewtonSolver,
    ):
        self.attractor = attractor
        self.visualizer_cls = visualizer
        self.solver: Solver = solver_cls(attractor)
        self.sim_settings = sim_settings
        if starting_state is None:
            starting_state = recommended_starting_states[type(self.attractor)].generate(
                self.sim_settings.num_particles
            )
        self.starting_state = starting_state
        recurrent_solver = RecurrentSolver(self.solver, starting_state, sim_settings.dt)
        self.buffered_solver = RingBufferedSolver(recurrent_solver, size_rb=10000)

    def run(self):
        if self.sim_settings.fast_start:
            self._pre_run()
        self.visualizer_cls(self.buffered_solver).visualize()

    def _pre_run(self):
        self.starting_state = self.solver.solve(self.starting_state, n_steps=10000, dt=0.01)[:, -1]

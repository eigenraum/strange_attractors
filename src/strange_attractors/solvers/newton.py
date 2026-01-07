import numpy as np

from strange_attractors.solvers.solver import Solver


class NewtonSolver(Solver):
    def solve(self, state: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        states = [state.copy()]
        for _ in range(n_steps - 1):
            prev = states[-1]
            delta_state = self._attractor.vector_field(prev)
            states.append(prev + delta_state * dt)
        return np.stack(states, axis=1)

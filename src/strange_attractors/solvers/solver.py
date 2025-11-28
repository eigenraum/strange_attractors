from abc import ABC, abstractmethod

import numpy as np

from strange_attractors.attractors import Attractor
from strange_attractors.utils.ringbuffer import TrajectoryBuffer


class Solver(ABC):
    def __init__(self, attractor: Attractor):
        self._attractor: Attractor = attractor

    @abstractmethod
    def solve(self, state: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        """Takes states of shape (n_samples, n_dimensions) and returns an array
        of shape (n_samples, n_steps, n_dimensions)

        """
        ...


class RecurrentSolver:
    """A recurrent solver class that emits solved trajectory snippets."""

    def __init__(self, solver: Solver, state: np.ndarray, dt: float):
        self._solver = solver
        self._state = state
        self._dt = dt

    @property
    def state(self):
        return self._state

    def next(self, n_steps: int):
        result = self._solver.solve(self._state, n_steps, self._dt)
        self._state = result[:, -1]
        return result


class RingBufferedSolver:
    """A Recurrent solver combined with a ring buffer.

    Computes a sequence of N new trajectory steps, and then outputs M>N
    trajectory slices, thereby keeping part of the old history.
    """

    def __init__(self, rec_solver: RecurrentSolver, size_rb: int, fill: bool = True):
        """
        Creates a ringbuffer of the specified size.
        If fill == True, will compute a first part of the solution to
        pre-fill the ring buffer.
        """
        self.rec_solver = rec_solver
        shape = (1, size_rb, 3)  # hard code for the moment
        self._rb = TrajectoryBuffer(tuple(shape))
        self.size_rb = size_rb
        if fill:
            self.update(size_rb)

    def get(self):
        return self._rb.get()

    def update(self, n_steps: int) -> np.ndarray:
        new_trajectories = self.rec_solver.next(n_steps)
        self._rb.append(new_trajectories)
        return self._rb.get()

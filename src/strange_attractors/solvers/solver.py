from abc import ABC, abstractmethod

import numpy as np

from strange_attractors.attractors import Attractor


class Solver(ABC):
    def __init__(self, attractor: Attractor):
        self._attractor: Attractor = attractor

    @abstractmethod
    def solve(self, state: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        """Takes states of shape (n_samples, n_dimensions) and returns an array
        of shape (n_samples, n_steps, n_dimensions)

        """
        ...

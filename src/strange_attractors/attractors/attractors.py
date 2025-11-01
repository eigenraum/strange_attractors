from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Attractor(ABC):
    @property
    @abstractmethod
    def n_dim(self) -> int:
        """
        Returns dimension of the attractor.
        """
        ...

    @abstractmethod
    def vector_field(self, vec: np.ndarray) -> np.ndarray:
        """Vector Field of Attractor.

        Input: np.ndarray shape (N, n_dim)
        Output: np.ndarray shape (N, n_dim)
        """
        ...


class SuperpositionAttractor(Attractor):
    def __init__(self, attractors: Sequence[Attractor]):
        self._attractors = attractors

    @property
    def n_dim(self) -> int:
        return self._attractors[0].n_dim

    def vector_field(self, vec: np.ndarray) -> np.ndarray:
        return np.sum(attractor.vector_field(vec) for attractor in self._attractors)  # type: ignore

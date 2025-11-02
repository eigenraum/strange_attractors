from dataclasses import dataclass

import numpy as np

from strange_attractors.attractors import Attractor


@dataclass(frozen=True)
class RosslerAttractor(Attractor):
    a: float = 0.2
    b: float = 0.2
    c: float = 5.7

    @property
    def n_dim(self) -> int:
        return 3

    def vector_field(self, vec: np.ndarray) -> np.ndarray:
        """Vector field for the Rössler attractor."""
        x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
        dx = -(y + z)
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return np.stack((dx, dy, dz), axis=1)

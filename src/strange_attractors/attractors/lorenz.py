from dataclasses import dataclass

import numpy as np

from strange_attractors.attractors import Attractor


@dataclass(frozen=True)
class LorenzAttractor(Attractor):
    a: float = 10.0
    b: float = 28.0
    c: float = 8.0 / 3

    @property
    def n_dim(self) -> int:
        return 3

    def vector_field(self, vec: np.ndarray) -> np.ndarray:
        """Vector Field of Lorenz Attractor"""
        x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
        dx = self.a * (y - x)
        dy = x * (self.b - z) - y
        dz = x * y - self.c * z
        return np.stack((dx, dy, dz), axis=1)

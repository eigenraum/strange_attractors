from dataclasses import dataclass

import numpy as np

from strange_attractors.attractors import Attractor


@dataclass(frozen=True)
class ThomasAttractor(Attractor):
    a: float = 0.19

    @property
    def n_dim(self) -> int:
        return 3

    def vector_field(self, vec: np.ndarray) -> np.ndarray:
        """Vector Field of Lorenz Attractor"""
        x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
        dx = -self.a * x + np.sin(y)
        dy = -self.a * y + np.sin(z)
        dz = -self.a * z + np.sin(x)
        return np.stack((dx, dy, dz), axis=1)

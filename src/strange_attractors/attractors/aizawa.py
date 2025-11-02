from dataclasses import dataclass

import numpy as np

from strange_attractors.attractors import Attractor


@dataclass(frozen=True)
class AizawaAttractor(Attractor):
    a: float = 0.95
    b: float = 0.7
    c: float = 0.6
    d: float = 3.5
    e: float = 0.25
    f: float = 0.1

    @property
    def n_dim(self) -> int:
        return 3

    def vector_field(self, vec: np.ndarray) -> np.ndarray:
        """Vector field for the Aizawa attractor."""
        x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
        dx = (z - self.b) * x - self.d * y
        dy = self.d * x + (z - self.b) * y
        dz = (
            self.c
            + self.a * z
            - (z ** 3) / 3.0
            - (x ** 2 + y ** 2) * (1 + self.f * z)
            + self.e * z
        )
        return np.stack((dx, dy, dz), axis=1)

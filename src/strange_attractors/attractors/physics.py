"""Physics attractors.

I have ideas for adding more physics attractors to simulate a pendulum that is affected by magnets.
These will however require the state space dimensionality to grow.
Need to think about how to implement this nicely while keeping a clean 2/3d visualization state space.
"""

from dataclasses import dataclass

import numpy as np

from strange_attractors.attractors import Attractor


@dataclass
class GravityAttractor(Attractor):
    """Gravity Attractor

    Applies 'force' to a system of dimension 'dims' in the dimension 'force_dim'.
    """

    force: float = -1.0
    dims: int = 3
    force_dim: int = 2

    @property
    def n_dim(self) -> int:
        return self.dims

    def vector_field(self, vec: np.ndarray) -> np.ndarray:
        force = np.zeros_like(vec)
        force[..., self.force_dim] = self.force
        return force

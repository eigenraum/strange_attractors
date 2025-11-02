from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from strange_attractors.attractors.lorenz import LorenzAttractor
from strange_attractors.attractors.thomas import ThomasAttractor


class StartingStates(ABC):
    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        pass


class RandomStartingStates(StartingStates):
    def __init__(self, ndim, randfunction=np.random.randn):
        self.n_dim = ndim
        self.randfunction = randfunction

    def generate(self, n: int) -> np.ndarray:
        return self.randfunction(n, self.n_dim)


class BoxStartingStates(StartingStates):
    def __init__(self, mins: Sequence[int], maxs: Sequence[int]):
        self.mins = [mins]
        self.maxs = [maxs]

    def generate(self, n: int) -> np.ndarray:
        return np.random.uniform(self.mins, self.maxs, size=(n, len(self.mins[0])))


# Appropriate states for Lorenz attractor
lorenz_box = BoxStartingStates([-10, -10, 10], [10, 10, 20])
randn_scaled = lambda *args, **kwargs: np.random.randn(*args, **kwargs) * 3.0
thomas_random = RandomStartingStates(3, randfunction=randn_scaled)

recommended_starting_states = defaultdict(lambda: thomas_random)
recommended_starting_states[type(LorenzAttractor)] = lorenz_box
recommended_starting_states[type(ThomasAttractor)] = thomas_random

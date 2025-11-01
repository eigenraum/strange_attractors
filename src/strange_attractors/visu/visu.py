from abc import ABC, abstractmethod

import numpy as np


class Visualizer(ABC):
    def __init__(self, trajectory: np.ndarray):
        self.trajectory = trajectory

    @abstractmethod
    def visualize(self): ...

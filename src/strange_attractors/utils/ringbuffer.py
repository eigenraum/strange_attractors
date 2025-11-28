import numpy as np


class TrajectoryBuffer:
    def __init__(self, shape: tuple[int, ...], ring_axis=1):
        self._trajectory = np.zeros(shape)
        self._ring_axis = ring_axis
        assert ring_axis == 1, "Other ring axes than 1 not supported"
        self._buffer_size = shape[ring_axis]

    def append(self, data: np.ndarray):
        num_items = data.shape[self._ring_axis]
        # move back part to front
        remaining = self._buffer_size - num_items
        self._trajectory[:, :remaining] = self._trajectory[:, num_items:]
        # Insert new back part
        self._trajectory[:, remaining:] = data

    def get(self) -> np.ndarray:
        return self._trajectory

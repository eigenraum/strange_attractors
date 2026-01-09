import numpy as np

from strange_attractors.utils.ringbuffer import TrajectoryBuffer


def test_ringbuffer():
    shape = (4, 10, 3)
    buffer = TrajectoryBuffer(shape, ring_axis=1)
    new_column = np.ones((4, 1, 3))
    buffer.append(new_column)
    new_column = np.ones((4, 2, 3)) * 2
    buffer.append(new_column)

    expected = np.zeros(shape)
    expected[:, -3] = 1
    expected[:, -2:] = 2

    np.testing.assert_allclose(buffer.get(), expected)

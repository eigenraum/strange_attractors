import numpy as np

from strange_attractors.attractors import LorenzAttractor


def test_lorenz_attractor():
    attractor = LorenzAttractor()
    positions = np.array([[0.0, 0.0, 0.0]])
    expected_vf = np.array([[0.0, 0.0, 0.0]])
    vf = attractor.vector_field(positions)
    np.testing.assert_allclose(vf, expected_vf)


def test_lorenz_dim():
    attractor = LorenzAttractor()
    expected_dim = 3
    assert attractor.n_dim == expected_dim

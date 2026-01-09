"""Reasonable default configurations for the Attractors"""

from strange_attractors.attractors import LorenzAttractor, ThomasAttractor
from strange_attractors.configs.attractor_config import AttractorConfig, SimSettings
from strange_attractors.visu.matplotlib import MatplotlibVisualizer3D
from strange_attractors.visu.vispy import VispyVisualizer3D

lorenz = AttractorConfig(
    attractor=LorenzAttractor(),
    visualizer=VispyVisualizer3D,
    sim_settings=SimSettings(
        dt=0.001,
        fast_start=True,
        num_particles=1,
        n_steps=50000,
        ring_buffer_size=100000,
        n_flow=10,
    ),
)

lorenz_single = AttractorConfig(
    attractor=LorenzAttractor(),
    visualizer=MatplotlibVisualizer3D,
    sim_settings=SimSettings(num_particles=1, ring_buffer_size=10000),
)

thomas = AttractorConfig(
    attractor=ThomasAttractor(),
    visualizer=VispyVisualizer3D,
    sim_settings=SimSettings(dt=0.03, num_particles=1, n_steps=50000, ring_buffer_size=100000),
)

thomas09 = AttractorConfig(
    attractor=ThomasAttractor(a=0.19),
    visualizer=VispyVisualizer3D,
    sim_settings=SimSettings(dt=0.03, num_particles=1, n_steps=50000, ring_buffer_size=10000),
)

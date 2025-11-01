"""Reasonable default configurations for the Attractors"""

from strange_attractors.attractors import LorenzAttractor, ThomasAttractor
from strange_attractors.configs.attractor_config import AttractorConfig, SimSettings
from strange_attractors.visu.matplotlib import MatplotlibVisualizer3D
from strange_attractors.visu.vispy import VispyVisualizer3D

lorenz = AttractorConfig(
    attractor=LorenzAttractor(),
    visualizer=VispyVisualizer3D,
    sim_settings=SimSettings(dt=0.001, fast_start=True),
)

lorenz_single = AttractorConfig(
    attractor=LorenzAttractor(),
    visualizer=MatplotlibVisualizer3D,
    sim_settings=SimSettings(num_particles=1),
)

thomas = AttractorConfig(
    attractor=ThomasAttractor(), visualizer=VispyVisualizer3D, sim_settings=SimSettings(dt=0.03)
)

thomas09 = AttractorConfig(
    attractor=ThomasAttractor(a=0.09),
    visualizer=VispyVisualizer3D,
    sim_settings=SimSettings(dt=0.03),
)

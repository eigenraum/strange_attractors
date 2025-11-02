"""Reasonable default configurations for the Attractors"""

from strange_attractors.attractors import (
    AizawaAttractor,
    LorenzAttractor,
    RosslerAttractor,
    ThomasAttractor,
)
from strange_attractors.configs.attractor_config import AttractorConfig, SimSettings
from strange_attractors.visu.matplotlib import MatplotlibVisualizer3D
from strange_attractors.visu.vispy import VispyVisualizer3D

lorenz = AttractorConfig(
    attractor=LorenzAttractor(),
    visualizer=VispyVisualizer3D,
    sim_settings=SimSettings(dt=0.001, fast_start=True, num_particles=1, n_steps=50000),
)

lorenz_single = AttractorConfig(
    attractor=LorenzAttractor(),
    visualizer=MatplotlibVisualizer3D,
    sim_settings=SimSettings(num_particles=1),
)

thomas = AttractorConfig(
    attractor=ThomasAttractor(),
    visualizer=VispyVisualizer3D,
    sim_settings=SimSettings(dt=0.03, num_particles=1, n_steps=50000),
)

thomas09 = AttractorConfig(
    attractor=ThomasAttractor(a=0.19),
    visualizer=VispyVisualizer3D,
    sim_settings=SimSettings(dt=0.03, num_particles=1, n_steps=50000),
)

rossler = AttractorConfig(
    attractor=RosslerAttractor(),
    visualizer=VispyVisualizer3D,
    sim_settings=SimSettings(dt=0.01, num_particles=1, n_steps=50000),
)

rossler_single = AttractorConfig(
    attractor=RosslerAttractor(),
    visualizer=MatplotlibVisualizer3D,
    sim_settings=SimSettings(dt=0.01, num_particles=1, n_steps=20000),
)

aizawa = AttractorConfig(
    attractor=AizawaAttractor(),
    visualizer=VispyVisualizer3D,
    sim_settings=SimSettings(dt=0.01, num_particles=1, n_steps=60000),
)

aizawa_single = AttractorConfig(
    attractor=AizawaAttractor(),
    visualizer=MatplotlibVisualizer3D,
    sim_settings=SimSettings(dt=0.01, num_particles=1, n_steps=20000),
)

"""Some quick and dirty code to run a series of experiments."""

from strange_attractors.attractors import ThomasAttractor
from strange_attractors.configs.configs import *
from strange_attractors.visu.vispy import VispyVisualizer3D

configs = [
    AttractorConfig(
        attractor=ThomasAttractor(a=a),
        visualizer=VispyVisualizer3D,
        sim_settings=SimSettings(dt=0.03, fast_start=True, n_steps=1000),
    )
    for a in (0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.19, 0.2)
]
config = AttractorConfig(
    attractor=ThomasAttractor(a=0.2),
    visualizer=VispyVisualizer3D,
    sim_settings=SimSettings(dt=0.03, fast_start=False, n_steps=3000),
)
config.run()
for config in configs:
    config.run()

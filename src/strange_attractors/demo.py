from strange_attractors.attractors import LorenzAttractor, ThomasAttractor
from strange_attractors.configs.configs import *
from strange_attractors.solvers.newton import NewtonSolver
from strange_attractors.utils.starting_states import recommended_starting_states
from strange_attractors.visu.matplotlib import MatplotlibVisualizer3D
from strange_attractors.visu.vispy import VispyVisualizer3D

configs = [
    AttractorConfig(
        attractor=ThomasAttractor(a=a),
        visualizer=VispyVisualizer3D,
        sim_settings=SimSettings(dt=0.03, fast_start=True),
    )
    for a in (0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15)
]
for config in configs:
    config.run()

thomas09.run()
lorenz.run()

attractor = LorenzAttractor()
attractor = ThomasAttractor()

# gravity = GravityAttractor(100.0)
# superposed_attractor = SuperpositionAttractor((attractor, gravity))
solver = NewtonSolver(attractor)

num_particles = 1000
starting_state = recommended_starting_states[type(attractor)].generate(num_particles)

fast_start = True
# Let's do a fast startup that is not visualized
if fast_start:
    trajectories = solver.solve(starting_state, n_steps=10000, dt=0.01)
    starting_state = trajectories[:, -1].copy()
    del trajectories
# Start visualizing from here
trajectories = solver.solve(starting_state, n_steps=10000, dt=0.01)

# Choose one (comment out the other)
if num_particles == 1:
    visualizer = MatplotlibVisualizer3D(trajectories)
else:
    visualizer = VispyVisualizer3D(trajectories)

visualizer.visualize()

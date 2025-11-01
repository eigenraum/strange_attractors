import numpy as np
from matplotlib import pyplot as plt

from strange_attractors.visu.visu import Visualizer


class MatplotlibVisualizer(Visualizer):
    def __init__(self, trajectory: np.ndarray):
        self.trajectory = trajectory

    def visualize(self):
        for t in self.trajectory:
            plt.plot(t[:, 0], t[:, 1])
        plt.show()


class MatplotlibVisualizer3D(Visualizer):
    def __init__(self, trajectory: np.ndarray, *, progress_color: bool = True):
        """Initialize a 3D visualizer.

        Args:
            trajectory (np.ndarray): The trajectory to visualize. Shape (n_particles, n_steps, 3)
            progress_color (bool, optional): Whether to use a color gradient to indicate progress. Defaults to True.
        """
        self.trajectory = trajectory
        self.progress_color = progress_color

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_args = {}
        if self.progress_color:
            index = np.linspace(0, 1, self.trajectory.shape[1])
            plot_args["c"] = index
            plot_args["cmap"] = "viridis"

        for t in self.trajectory:
            ax.scatter(t[:, 0], t[:, 1], t[:, 2], **plot_args, marker=".")
        mins = self.trajectory.reshape(-1, 3).min(axis=0)
        maxs = self.trajectory.reshape(-1, 3).max(axis=0)
        ax.set_box_aspect(maxs - mins)  # proper 3D proportions
        ax.view_init(elev=30, azim=45)  # more obvious 3D view
        ax.set_xlabel("X", labelpad=8)
        ax.set_ylabel("Y", labelpad=8)
        ax.set_zlabel("Z", labelpad=8)

        plt.show()

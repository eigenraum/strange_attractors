# pip install vispy
import numpy as np
from matplotlib.cm import get_cmap
from vispy import app, scene

from strange_attractors.visu.visu import Visualizer


class VispyVisualizer3D(Visualizer):
    def __init__(
        self,
        trajectory: np.ndarray,  # (n_particles, n_steps, 3)
        *,
        use_color: bool = True,  # ignored (no coloring)
        fps: int = 120,
        trail_steps: int = 100,  # 0 => only current positions; >0 => short trails
        point_size: float = 1.5,
        background: str = "black",
    ):
        self.trajectory = np.asarray(trajectory, dtype=np.float32)
        if self.trajectory.ndim != 3 or self.trajectory.shape[2] != 3:
            raise ValueError("trajectory must have shape (n_particles, n_steps, 3)")
        self.fps = int(fps)
        self.trail_steps = int(trail_steps)
        self.point_size = float(point_size)
        self.background = background
        self.cmap = get_cmap("Reds")

        if use_color:
            N = self.trajectory.shape[0]
            rng = np.random.default_rng(0)
            self.colors = np.c_[rng.random((N, 3)), np.ones(N, dtype=np.float32)].astype(np.float32)
        else:
            self.colors = "white"

    def visualize(self):
        N, T, _ = self.trajectory.shape

        # Canvas + 3D view
        canvas = scene.SceneCanvas(
            keys="interactive", size=(900, 700), bgcolor=self.background, show=True
        )
        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=45, elevation=30, up="+z")

        # Visual: start at frame 0
        scatter = scene.visuals.Markers(parent=view.scene)
        scatter.set_data(
            self.trajectory[:, 0, :], face_color=self.colors, size=self.point_size, edge_width=0
        )

        # Fit camera to full data bounds (all times), like set_box_aspect
        mins = self.trajectory.reshape(-1, 3).min(axis=0)
        maxs = self.trajectory.reshape(-1, 3).max(axis=0)
        view.camera.set_range(x=(mins[0], maxs[0]), y=(mins[1], maxs[1]), z=(mins[2], maxs[2]))

        # Animation state
        state = {"frame": 0, "play": True}

        def render_frame(i: int):
            speed = self.trajectory[:, i - 1, :] - self.trajectory[:, i, :]
            speed_mag = np.linalg.norm(speed, axis=1).astype(np.float32)
            w = (speed_mag - speed_mag.min()) / (np.ptp(speed_mag) + 1e-12)
            # self.colors = np.c_[w, w, w, np.ones_like(w, dtype=np.float32)]
            self.colors = self.cmap(w)

            if self.trail_steps > 0:
                a = max(0, i - self.trail_steps + 1)
                seg = self.trajectory[:, a : i + 1, :].reshape(-1, 3)  # (n_particles * trail, 3)
                if isinstance(self.colors, np.ndarray):
                    k = i - a + 1
                    colors_rep = np.repeat(self.colors, k, axis=0)
                    fade = np.logspace(-2.0, 0, k, dtype=np.float32)  # oldest → newest
                    w = np.tile(fade, self.trajectory.shape[0])
                    colors_rep[:, 3] *= w  # fade via alpha

                else:  # is string
                    colors_rep = self.colors
                scatter.set_data(
                    seg,
                    face_color=colors_rep,
                    size=self.point_size,
                    edge_width=0,
                )
                scatter.order = "translucent"
                scatter.set_gl_state(blend=True, depth_test=False, depth_mask=False)

                # scatter.set_data(seg, face_color=self.colors, size=self.point_size, edge_width=0)
            else:
                scatter.set_data(
                    self.trajectory[:, i, :],
                    face_color=self.colors,
                    size=self.point_size,
                    edge_width=0,
                )

        def on_timer(event):
            if not state["play"]:
                return
            state["frame"] = (state["frame"] + 1) % T
            render_frame(state["frame"])

        # Timer at target FPS
        timer = app.Timer(interval=1.0 / max(self.fps, 1), connect=on_timer, start=True)

        # Basic controls: Space pause/resume, arrows step
        @canvas.events.key_press.connect
        def on_key(event):
            if event.key == "Space":
                state["play"] = not state["play"]
            elif event.key == "Right":
                state["frame"] = (state["frame"] + 1) % T
                render_frame(state["frame"])
            elif event.key == "Left":
                state["frame"] = (state["frame"] - 1) % T
                render_frame(state["frame"])

        app.run()

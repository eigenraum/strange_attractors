"""Real-time 3D visualization using VisPy."""

import contextlib

import imageio as iio
import numpy as np
from matplotlib.cm import get_cmap
from vispy import app, scene

from strange_attractors.solvers.solver import RingBufferedSolver
from strange_attractors.visu.visu import Visualizer


class VispyVisualizer3D(Visualizer):
    """Real-time 3D visualizer using a RingBufferedSolver for live trajectory updates."""

    def __init__(
        self,
        solver: RingBufferedSolver,
        *,
        fps: int = 60,
        steps_per_frame: int = 10,
        point_size: float = 1.5,
        background: str = "black",
        cmap: str = "inferno",
        output: str | None = None,
        n_flow: int = 10,
    ):
        """
        Args:
            solver: RingBufferedSolver that provides trajectory data.
            fps: Target frames per second.
            steps_per_frame: Number of new trajectory steps computed per frame cycle (after n_flow frames).
            point_size: Size of rendered points.
            background: Background color.
            cmap: Matplotlib colormap name for velocity coloring.
            output: Path to output video file (e.g., "output.mp4"), or None for no recording.
            n_flow: Number of frames in the flow cycle. Trajectory is subsampled as [offset::n_flow] where offset cycles 0 to n_flow-1.
        """
        self.solver = solver
        self.fps = fps
        self.steps_per_frame = steps_per_frame
        self.point_size = point_size
        self.background = background
        self.cmap = get_cmap(cmap)
        self.output = output
        self.n_flow = n_flow

        # Track all-time speed range (only expands, never shrinks)
        self.speed_min = float("inf")
        self.speed_max = float("-inf")

        # Track current offset in the flow cycle
        self.flow_offset = 0

    def _compute_colors(self, trajectory: np.ndarray, offset: int) -> np.ndarray:
        """Compute colors based on velocity magnitude with fading trail effect.

        Args:
            trajectory: Shape (n_particles, n_steps, 3) - the full dense trajectory
            offset: Current offset in the flow cycle (0 to n_flow-1)

        Returns:
            RGBA colors of shape (n_particles * subsampled_steps, 4) for trajectory[offset::n_flow]
        """
        n_particles, n_steps, _ = trajectory.shape

        # Subsample the trajectory
        subsampled = trajectory[:, offset::self.n_flow, :]
        n_sub_steps = subsampled.shape[1]

        # Compute velocity (difference between consecutive steps in the ORIGINAL trajectory)
        # We want colors based on actual velocity, not subsampled velocity
        velocity = np.diff(trajectory, axis=1)
        speed = np.linalg.norm(velocity, axis=-1)  # (n_particles, n_steps-1)

        # Pad to match trajectory length (repeat first speed)
        speed = np.concatenate([speed[:, :1], speed], axis=1)  # (n_particles, n_steps)

        # Subsample the speed to match the subsampled trajectory
        speed_sub = speed[:, offset::self.n_flow]

        # Update all-time speed range (only expands)
        self.speed_min = min(self.speed_min, speed.min())
        self.speed_max = max(self.speed_max, speed.max())

        # Normalize speed for colormap using stable all-time range
        speed_norm = (speed_sub - self.speed_min) / (self.speed_max - self.speed_min + 1e-12)

        # Apply colormap (returns RGBA with A=1.0)
        colors = np.array(self.cmap(speed_norm), dtype=np.float32)  # (n_particles, n_sub_steps, 4)

        # Apply fading trail: exponential decay from 0.1% (oldest) to 100% (newest)
        # Index 0 = oldest point = 0.001 opacity
        # Index -1 = newest point = 1.0 opacity
        fade = np.logspace(-3, 0, n_sub_steps, dtype=np.float32)
        colors[:, :, 3] = fade  # Set alpha directly (not multiply)

        return colors.reshape(-1, 4)

    def visualize(
        self,
        *,
        size: tuple[int, int] = (904, 704),
        show: bool = True,
        limit_frames: int | None = None,
        codec: str = "libx264",
        quality: int = 8,
    ):
        """Run the live visualization.

        Args:
            size: Canvas size (width, height).
            show: Whether to show the window.
            limit_frames: Stop after this many frames (for recording). None = run forever.
            codec: Video codec for recording.
            quality: Video quality (0-10).
        """
        # Get initial trajectory for camera setup
        trajectory = self.solver.get()

        # Create canvas and view
        canvas = scene.SceneCanvas(
            keys="interactive", size=size, bgcolor=self.background, show=show
        )
        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=135, elevation=35, up="+z")

        # Initialize scatter plot
        scatter = scene.visuals.Markers(parent=view.scene)

        # Set camera range based on initial trajectory
        points = trajectory.reshape(-1, 3)
        mins, maxs = points.min(axis=0), points.max(axis=0)
        view.camera.set_range(x=(mins[0], maxs[0]), y=(mins[1], maxs[1]), z=(mins[2], maxs[2]))

        # Animation state
        state = {"paused": False, "rotate": False, "frame_count": 0}

        # Video writer
        writer = None
        if self.output is not None:
            writer = iio.get_writer(
                self.output,
                fps=self.fps,
                codec=codec,
                quality=quality,
                pixelformat="yuv420p",
            )

        def render():
            """Render the current trajectory state with subsampling based on flow offset."""
            traj = self.solver.get()

            # Subsample trajectory based on current flow offset
            subsampled_traj = traj[:, self.flow_offset::self.n_flow, :]
            colors = self._compute_colors(traj, self.flow_offset)
            points = subsampled_traj.reshape(-1, 3)

            scatter.set_data(points, face_color=colors, size=self.point_size, edge_width=0)
            scatter.set_gl_state(blend=True, depth_test=False, depth_mask=False)

        def on_timer(_):
            if state["paused"]:
                return

            # Advance flow offset
            self.flow_offset = (self.flow_offset + 1) % self.n_flow

            # Only compute new trajectory points after completing a full flow cycle
            if self.flow_offset == 0:
                self.solver.update(self.steps_per_frame)

            # Render
            render()

            # Rotate camera if enabled
            if state["rotate"]:
                view.camera.azimuth += 0.1

            canvas.update()

            # Record frame if writer is active
            if writer is not None:
                img = canvas.render()
                writer.append_data(img)
                state["frame_count"] += 1

                if limit_frames is not None and state["frame_count"] >= limit_frames:
                    writer.close()
                    app.quit()

        # Start timer (must keep reference to prevent garbage collection)
        _timer = app.Timer(interval=1.0 / self.fps, connect=on_timer, start=True)

        @canvas.events.key_press.connect
        def on_key(event):
            if event.key == "Space":
                state["paused"] = not state["paused"]
            elif event.key == "R":
                state["rotate"] = not state["rotate"]
            elif event.key == "Up":
                self.steps_per_frame = min(self.steps_per_frame + 5, 100)
            elif event.key == "Down":
                self.steps_per_frame = max(self.steps_per_frame - 5, 1)
            elif event.key == "C":
                cam = view.camera
                print(f"[cam] az={cam.azimuth:.2f}°, el={cam.elevation:.2f}°, fov={cam.fov:.1f}")

        @canvas.events.close.connect
        def on_close(_):
            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.close()

        # Initial render
        render()
        app.run()

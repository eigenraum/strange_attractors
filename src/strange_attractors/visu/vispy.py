"""
AI generated mess, to be cleaned up.
"""

import imageio as iio
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
        fps: int = 60,
        trail_steps: int = 50000,  # 0 => only current positions; >0 => short trails
        point_size: float = 1.5,
        background: str = "black",
        output: str | None = None,  # Give it a filename.mp4 to record video.
        frame_steps: int = 10,
    ):
        self.trajectory = np.asarray(trajectory, dtype=np.float32)
        if self.trajectory.ndim != 3 or self.trajectory.shape[2] != 3:
            raise ValueError("trajectory must have shape (n_particles, n_steps, 3)")
        self.fps = int(fps)
        self.trail_steps = int(trail_steps)
        self.point_size = float(point_size)
        self.background = background
        self.cmap = get_cmap("Reds")
        self.output = output
        self.frame_steps = frame_steps

        if use_color:
            N = self.trajectory.shape[0]
            rng = np.random.default_rng(0)
            self.colors = np.c_[rng.random((N, 3)), np.ones(N, dtype=np.float32)].astype(np.float32)
            velocity = self.trajectory[:, :-1, :] - self.trajectory[:, 1:, :]
            speed = np.linalg.norm(velocity, axis=-1)
            self.speed_min = np.min(speed)
            self.speed_max = np.max(speed)
            speed_norm = (speed - self.speed_min) / (self.speed_max - self.speed_min + 1e-12)
            speed_norm = np.concatenate((speed_norm[:, 0:1], speed_norm), axis=1)
            self.color = self.cmap(speed_norm)
        else:
            self.colors = "white"

    def visualize(
        self,
        *,
        size=(904, 704),
        show=True,
        # If self.output is not None, choose one of these:
        loops: int = 1,  # how many times to cycle through the trajectory
        limit_frames: int | None = None,  # or record a fixed #frames
        codec: str = "libx264",
        quality: int = 8,
        pixelformat: str = "yuv420p",
        alpha: bool = False,  # keep as False for video encoders
    ):
        """
        Play the animation; optionally record it to a video file.

        Args:
            size: output canvas size in (W,H) pixels (also the video resolution).
            show: True to show a window; False for headless (needs appropriate backend/driver).
            loops: how many full cycles of the trajectory to record (used if limit_frames is None).
            limit_frames: record exactly this many frames (overrides loops if provided).
            codec, quality, pixelformat: ffmpeg writer options.
            alpha: pass True to capture RGBA; most codecs ignore A, so default False.
        """
        N, T, _ = self.trajectory.shape

        # Canvas + 3D view
        canvas = scene.SceneCanvas(
            keys="interactive", size=size, bgcolor=self.background, show=show
        )
        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=135, elevation=35, up="+z")

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
        state = {"frame": 0, "play": True, "move": False}

        # ---- optional video writer ----
        writer = None
        frames_target = None
        frames_written = 0

        record_to = self.output
        if record_to is not None:
            # If limit_frames not provided, compute from loops
            frames_target = limit_frames if limit_frames is not None else int(loops * T)
            # Ensure canvas has the requested output size (W,H)
            canvas.size = size

            writer = iio.get_writer(
                record_to,
                fps=max(self.fps, 1),
                codec=codec,
                pixelformat=pixelformat,
                quality=int(np.clip(quality, 0, 10)),
            )

        def render_frame(i: int):
            # Color by per-particle speed (between frames i-1 and i)
            speed = self.trajectory[:, i - 1, :] - self.trajectory[:, i, :]
            speed_mag = np.linalg.norm(speed, axis=1).astype(np.float32)
            w = (speed_mag - self.speed_min) / (self.speed_max - self.speed_min + 1e-12)
            colors_now = self.cmap(w)  # RGBA in [0,1]

            if self.trail_steps > 0:
                a = max(0, i - self.trail_steps + 1)
                seg = self.trajectory[:, a : i + 1, :].reshape(-1, 3)  # (n_particles * trail, 3)
                k = i - a + 1
                # colors_rep = np.repeat(colors_now, k, axis=0)
                colors_rep = self.color[:, a : i + 1, :].reshape(-1, 4).copy()
                fade = np.logspace(-1.5, 0, k, dtype=np.float32)  # oldest → newest
                wfade = np.tile(fade, self.trajectory.shape[0])
                colors_rep[:, 3] *= wfade  # fade via alpha

                scatter.set_data(
                    seg,
                    face_color=colors_rep,
                    size=self.point_size,
                    edge_width=0,
                )
                scatter.order = "translucent"
                scatter.set_gl_state(blend=True, depth_test=False, depth_mask=False)
            else:
                scatter.set_data(
                    self.trajectory[:, i, :],
                    face_color=colors_now,
                    size=self.point_size,
                    edge_width=0,
                )

        def capture_and_maybe_stop():
            nonlocal frames_written
            if writer is None:
                return False  # not stopping

            # Render the current frame to an image and write it out
            img = canvas.render(alpha=alpha)  # (H, W, 4) uint8 if alpha=True else (H,W,3/4)
            img = np.flipud(img)  # OpenGL origin flip

            # Some vispy backends return RGBA even if alpha=False; writer will accept it.
            writer.append_data(img)
            frames_written += 1

            if frames_target is not None and frames_written >= frames_target:
                # Finish recording and exit app
                writer.close()
                canvas.close()
                app.quit()
                return True
            return False

        def on_timer(_):
            if not state["play"]:
                return
            # advance to next frame
            state["frame"] = (state["frame"] + self.frame_steps) % T
            render_frame(state["frame"])
            # capture after drawing the new frame
            if capture_and_maybe_stop():
                return

            if state["move"]:
                view.camera.elevation += 0.1
            canvas.update()

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
                capture_and_maybe_stop()
                canvas.update()
            elif event.key == "Left":
                state["frame"] = (state["frame"] - 1) % T
                render_frame(state["frame"])
                capture_and_maybe_stop()
                canvas.update()
            elif event.key == "Up":
                self.frame_steps += 1
            elif event.key == "Down":
                self.frame_steps -= 1
            elif event.key == "M":
                state["move"] = not state["move"]

            elif event.key == "C":
                cam = view.camera
                print(f"[cam] az={cam.azimuth:.2f}°, el={cam.elevation:.2f}°, fov={cam.fov:.1f}")

        # Ensure writer closes if the window is closed early
        @canvas.events.close.connect
        def _on_close(_):
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass

        app.run()

"""Real-time 3D visualization using VisPy."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import imageio as iio
import numpy as np
from matplotlib.cm import get_cmap
from vispy import app, scene

from strange_attractors.solvers.solver import RingBufferedSolver
from strange_attractors.visu.visu import Visualizer

_HELP_PROMPT = "[h] help"
_HELP_POSITION = (12, 20)
_HELP_LINE_HEIGHT = 17
_HELP_FONT_SIZE = 11
_HELP_COLOR = (1.0, 1.0, 1.0, 0.82)
_POINT_SIZE_STEP = 0.25
_POINT_SIZE_MIN = 0.25
_POINT_SIZE_MAX = 20.0


@dataclass(frozen=True)
class KeyBinding:
    """Interactive keybinding plus the exact help text shown to the user."""

    key: str
    label: str
    description: str
    action: Callable[[], None]
    aliases: tuple[str, ...] = ()

    @property
    def normalized_keys(self) -> tuple[str, ...]:
        return tuple(key.lower() for key in (self.key, *self.aliases))


@dataclass
class HelpOverlay:
    """Top-left help overlay made from single-line Text nodes."""

    canvas: scene.SceneCanvas
    lines: list[scene.visuals.Text]

    def set_lines(self, lines: tuple[str, ...]) -> None:
        while len(self.lines) < len(lines):
            self.lines.append(_create_help_line(self.canvas, len(self.lines)))

        for index, text in enumerate(lines):
            line = self.lines[index]
            line.text = text
            line.visible = True

        for line in self.lines[len(lines) :]:
            line.visible = False


def _key_name(key: object) -> str | None:
    """Return VisPy's normalized key name for dispatching."""
    if key is None:
        return None

    return str(getattr(key, "name", key)).lower()


def _keybinding_index(key_bindings: tuple[KeyBinding, ...]) -> dict[str, KeyBinding]:
    bindings_by_key: dict[str, KeyBinding] = {}
    for binding in key_bindings:
        for key in binding.normalized_keys:
            if key in bindings_by_key:
                raise ValueError(f"duplicate keybinding for {binding.key!r}")
            bindings_by_key[key] = binding
    return bindings_by_key


def _find_key_binding(
    key_bindings_by_key: dict[str, KeyBinding],
    key: object,
    text: str | None = None,
) -> KeyBinding | None:
    candidates = (_key_name(key), text.lower() if text else None)
    for candidate in candidates:
        if candidate is not None and candidate in key_bindings_by_key:
            return key_bindings_by_key[candidate]
    return None


def _format_keybinding_help(key_bindings: tuple[KeyBinding, ...]) -> tuple[str, ...]:
    label_width = max(len(binding.label) for binding in key_bindings)
    lines = ["Keybindings"]
    lines.extend(
        f"{binding.label:<{label_width}}  {binding.description}" for binding in key_bindings
    )
    return tuple(lines)


def _create_help_line(canvas: scene.SceneCanvas, index: int) -> scene.visuals.Text:
    x, y = _HELP_POSITION
    return scene.visuals.Text(
        "",
        parent=canvas.scene,
        pos=(x, y + index * _HELP_LINE_HEIGHT),
        anchor_x="left",
        anchor_y="top",
        color=_HELP_COLOR,
        font_size=_HELP_FONT_SIZE,
    )


def _create_help_overlay(canvas: scene.SceneCanvas) -> HelpOverlay:
    overlay = HelpOverlay(canvas=canvas, lines=[_create_help_line(canvas, 0)])
    overlay.set_lines((_HELP_PROMPT,))
    return overlay


def _update_help_overlay(
    help_overlay: HelpOverlay,
    key_bindings: tuple[KeyBinding, ...],
    show_help: bool,
) -> None:
    lines = _format_keybinding_help(key_bindings) if show_help else (_HELP_PROMPT,)
    help_overlay.set_lines(lines)


def _set_camera_range(view: scene.widgets.ViewBox, trajectory: np.ndarray) -> None:
    points = trajectory.reshape(-1, 3)
    mins, maxs = points.min(axis=0), points.max(axis=0)
    view.camera.set_range(x=(mins[0], maxs[0]), y=(mins[1], maxs[1]), z=(mins[2], maxs[2]))


class VispyVisualizer3D(Visualizer):
    """Real-time 3D visualizer using a RingBufferedSolver for live trajectory updates."""

    def __init__(  # noqa: PLR0913
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
            steps_per_frame: Number of trajectory steps computed per flow cycle.
            point_size: Size of rendered points.
            background: Background color.
            cmap: Matplotlib colormap name for velocity coloring.
            output: Path to output video file (e.g., "output.mp4"), or None for no recording.
            n_flow: Number of frames in the flow cycle. Trajectory is subsampled
                as [offset::n_flow] where offset cycles from 0 to n_flow - 1.
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

    def _create_video_writer(self, codec: str, quality: int):
        if self.output is None:
            return None

        return iio.get_writer(
            self.output,
            fps=self.fps,
            codec=codec,
            quality=quality,
            pixelformat="yuv420p",
        )

    def _create_canvas(
        self,
        *,
        size: tuple[int, int],
        show: bool,
    ) -> tuple[scene.SceneCanvas, scene.widgets.ViewBox]:
        canvas = scene.SceneCanvas(size=size, bgcolor=self.background, show=show)
        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(
            fov=45,
            azimuth=135,
            elevation=35,
            up="+z",
        )
        return canvas, view

    def _render(self, scatter: scene.visuals.Markers) -> None:
        """Render the current trajectory state with flow-cycle subsampling."""
        traj = self.solver.get()

        subsampled_traj = traj[:, self.flow_offset::self.n_flow, :]
        colors = self._compute_colors(traj, self.flow_offset)
        points = subsampled_traj.reshape(-1, 3)

        scatter.set_data(points, face_color=colors, size=self.point_size, edge_width=0)
        scatter.set_gl_state(blend=True, depth_test=False, depth_mask=False)

    def _make_key_bindings(
        self,
        *,
        canvas: scene.SceneCanvas,
        view: scene.widgets.ViewBox,
        scatter: scene.visuals.Markers,
        help_overlay: HelpOverlay,
        state: dict[str, bool | int],
    ) -> tuple[KeyBinding, ...]:
        def rerender() -> None:
            self._render(scatter)
            canvas.update()

        def toggle_help() -> None:
            state["show_help"] = not state["show_help"]
            _update_help_overlay(help_overlay, key_bindings, bool(state["show_help"]))
            canvas.update()

        def toggle_pause() -> None:
            state["paused"] = not state["paused"]

        def toggle_rotation() -> None:
            state["rotate"] = not state["rotate"]

        def increase_steps_per_frame() -> None:
            self.steps_per_frame = min(self.steps_per_frame + 5, 100)

        def decrease_steps_per_frame() -> None:
            self.steps_per_frame = max(self.steps_per_frame - 5, 1)

        def increase_particle_size() -> None:
            self.point_size = min(self.point_size + _POINT_SIZE_STEP, _POINT_SIZE_MAX)
            rerender()

        def decrease_particle_size() -> None:
            self.point_size = max(self.point_size - _POINT_SIZE_STEP, _POINT_SIZE_MIN)
            rerender()

        def print_camera_position() -> None:
            cam = view.camera
            print(
                f"[cam] az={cam.azimuth:.2f}°, el={cam.elevation:.2f}°, "
                f"fov={cam.fov:.1f}"
            )

        def toggle_fullscreen() -> None:
            canvas.fullscreen = not canvas.fullscreen

        key_bindings = (
            KeyBinding("H", "[h]", "show/hide this help", toggle_help),
            KeyBinding("Space", "[space]", "pause/resume simulation", toggle_pause),
            KeyBinding("R", "[r]", "toggle camera rotation", toggle_rotation),
            KeyBinding("Up", "[up]", "increase steps per frame", increase_steps_per_frame),
            KeyBinding("Down", "[down]", "decrease steps per frame", decrease_steps_per_frame),
            KeyBinding("+", "[+]", "increase particle size", increase_particle_size),
            KeyBinding("-", "[-]", "decrease particle size", decrease_particle_size),
            KeyBinding("C", "[c]", "print camera position", print_camera_position),
            KeyBinding("F11", "[f11]", "toggle full screen", toggle_fullscreen),
            KeyBinding("Escape", "[esc]", "close visualization", canvas.close),
        )
        _update_help_overlay(help_overlay, key_bindings, bool(state["show_help"]))
        return key_bindings

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
        canvas, view = self._create_canvas(size=size, show=show)

        # Initialize scatter plot
        scatter = scene.visuals.Markers(parent=view.scene)
        help_overlay = _create_help_overlay(canvas)

        # Set camera range based on initial trajectory
        _set_camera_range(view, trajectory)

        # Animation state
        state: dict[str, bool | int] = {
            "paused": False,
            "rotate": False,
            "frame_count": 0,
            "show_help": False,
        }

        # Video writer
        writer = self._create_video_writer(codec, quality)
        key_bindings = self._make_key_bindings(
            canvas=canvas,
            view=view,
            scatter=scatter,
            help_overlay=help_overlay,
            state=state,
        )
        key_bindings_by_key = _keybinding_index(key_bindings)

        def on_timer(_):
            if state["paused"]:
                return

            # Advance flow offset
            self.flow_offset = (self.flow_offset + 1) % self.n_flow

            # Only compute new trajectory points after completing a full flow cycle
            if self.flow_offset == 0:
                self.solver.update(self.steps_per_frame)

            # Render
            self._render(scatter)

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
            binding = _find_key_binding(
                key_bindings_by_key,
                event.key,
                getattr(event, "text", None),
            )
            if binding is not None:
                binding.action()

        @canvas.events.close.connect
        def on_close(_):
            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.close()

        # Initial render
        self._render(scatter)
        app.run()

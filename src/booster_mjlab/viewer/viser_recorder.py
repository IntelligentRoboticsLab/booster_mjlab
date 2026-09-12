"""Viser play viewer with in-browser video recording.

mjlab's ``--video`` flag records an offscreen MuJoCo camera for a fixed number
of steps. This viewer instead records exactly what you see in the viser tab:
a ``Recording`` panel with a record / stop button that pulls frames from the
clicking client's browser (``ClientHandle.get_render``).

Two capture modes, chosen in the panel.

Lockstep drives the loop instead of sampling it: for each video frame it
advances the sim to exactly that frame's sim time, pushes the pose, and blocks
on the render, so every frame is distinct and uniformly spaced. The sim runs in
slow motion on screen while recording. Since the sim only produces a distinct
pose every ``step_dt`` (50Hz at the usual 0.005s timestep x 4 decimation), a
60fps video needs poses between control steps; lockstep interpolates body
transforms rather than repeating frames.

Realtime samples the browser as fast as it will go while the sim runs at normal
speed, duplicating frames to fill the timeline. The requested rate is a
container rate, not a distinct-pose rate, and the irregular duplication reads as
judder. Use it for long captures or when interacting matters more than
smoothness.

Both modes place frames on the video timeline by *simulated* time, so a pause
leaves a cut rather than a frozen stretch and the viewer's speed control records
deliberate slow motion.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import mediapy as media
import numpy as np
import viser
import viser.transforms as vtf
from mjlab.viewer.viser.viewer import ViserPlayViewer
from typing_extensions import override

DEFAULT_VIDEO_DIR = Path("logs/videos/viser")

# Label -> (height, width). Requests larger than the browser viewport get
# clamped by the client, so the written video may be smaller than asked for.
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480p": (480, 854),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}

LOCKSTEP = "Lockstep (exact fps)"
REALTIME = "Realtime (as shown)"

# Bound how long a single frame request may block, so stopping stays responsive.
_RENDER_TIMEOUT_S = 5.0
# How long to nap at most while waiting for the next frame to come due.
_POLL_CAP_S = 0.05
# Don't push huge files back through the websocket to the browser.
_MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
_SLERP_EPS = 1e-6


def _slerp(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc slerp between batches of wxyz quaternions."""
    dot = np.sum(qa * qb, axis=-1, keepdims=True)
    # q and -q are the same rotation; pick the sign that takes the short way.
    qb = np.where(dot < 0.0, -qb, qb)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    small = sin_theta < _SLERP_EPS
    safe = np.where(small, 1.0, sin_theta)
    wa = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    wb = np.where(small, t, np.sin(t * theta) / safe)
    out = wa * qa + wb * qb
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


@dataclass
class _SimSample:
    """Everything the viser scene needs to draw one instant of the sim."""

    sim_time: float
    xpos: np.ndarray
    xquat: np.ndarray
    mocap_pos: np.ndarray | None
    mocap_quat: np.ndarray | None
    qpos: np.ndarray | None
    qvel: np.ndarray | None
    ctrl: np.ndarray | None


@dataclass
class _Lockstep:
    """Mutable state of an in-progress lockstep recording."""

    client: viser.ClientHandle
    path: Path
    height: int
    width: int
    fps: float
    prev: _SimSample
    next: _SimSample
    # Advanced by the speed multiplier each frame, so a mid-recording speed
    # change bends the slope instead of jumping.
    sim_target: float = 0.0
    frames_written: int = 0
    last_step_count: int = 0
    writer: media.VideoWriter | None = None
    shape: tuple[int, int] | None = None
    last_frame: np.ndarray | None = None
    dropped: int = 0
    wall_start: float = field(default_factory=time.perf_counter)
    last_status: float = 0.0


class RecordingViserPlayViewer(ViserPlayViewer):
    """``ViserPlayViewer`` plus a record / stop-recording button."""

    def __init__(self, *args, video_dir: Path | str = DEFAULT_VIDEO_DIR, **kwargs):
        super().__init__(*args, **kwargs)
        self._video_dir = Path(video_dir)
        self._record_thread: threading.Thread | None = None
        self._record_stop = threading.Event()
        self._record_stop_client: viser.ClientHandle | None = None
        self._record_path: Path | None = None
        self._record_scene_boost = False
        self._lockstep: _Lockstep | None = None

    @override
    def setup(self) -> None:
        super().setup()
        self._setup_recording_gui()

    def _setup_recording_gui(self) -> None:
        # The tab group is local to the base setup(), so the recording controls
        # go at the root of the GUI panel, below the tabs.
        with self._server.gui.add_folder("Recording"):
            self._record_button = self._server.gui.add_button(
                "Record",
                icon=viser.Icon.PLAYER_RECORD,
                color="red",
                hint=(
                    "Capture this browser view to an mp4, timed by the sim clock "
                    "so it plays back in real time. Keep the tab visible: "
                    "backgrounded tabs stop producing frames."
                ),
            )
            self._record_mode = self._server.gui.add_dropdown(
                "Mode",
                options=(LOCKSTEP, REALTIME),
                initial_value=LOCKSTEP,
                hint=(
                    "Lockstep drives the sim one video frame at a time: every "
                    "frame is distinct and evenly spaced, but the sim runs in "
                    "slow motion while recording. Realtime keeps the sim at "
                    "normal speed and duplicates frames to fill the timeline."
                ),
            )
            self._record_resolution = self._server.gui.add_dropdown(
                "Resolution",
                options=tuple(RESOLUTIONS),
                initial_value="720p",
            )
            self._record_fps = self._server.gui.add_number(
                "FPS", initial_value=60, min=5, max=60, step=5
            )
            self._record_download = self._server.gui.add_checkbox(
                "Download when done",
                initial_value=True,
                hint="Send the finished mp4 to your browser (useful on a remote host).",
            )
            self._record_status = self._server.gui.add_html("")

        native = 1.0 / self._step_dt if self._step_dt else 0.0
        self._set_record_status(
            f"Idle. Videos are written to <code>{self._video_dir}</code>.<br/>"
            f"Sim produces {native:.0f} distinct poses/s; above that, lockstep "
            "interpolates."
        )

        @self._record_button.on_click
        def _(event: viser.GuiEvent) -> None:
            if not self._is_recording:
                self._start_recording(event.client)
            elif self._lockstep is not None:
                # Lockstep owns the main loop thread, likely blocked in
                # get_render; let it close its own writer.
                self._record_stop_client = event.client
                self._record_stop.set()
                self._set_record_status("Finishing recording&hellip;")
            else:
                self._stop_recording(event.client)

    @property
    def _is_recording(self) -> bool:
        if self._lockstep is not None:
            return True
        return self._record_thread is not None and self._record_thread.is_alive()

    @property
    def _step_dt(self) -> float:
        return float(getattr(self.env.unwrapped, "step_dt", 0.0) or 0.0)

    @property
    def _base_scene_hz(self) -> float:
        """Distinct poses per wall-second the base viewer sends to the browser."""
        return self.frame_rate / 2.0

    def _set_record_status(self, html: str) -> None:
        self._record_status.content = (
            '<div style="font-size: 0.85em; line-height: 1.25; '
            f'padding: 0 1em 0.5em 1em;">{html}</div>'
        )

    def _set_controls_disabled(self, disabled: bool) -> None:
        for handle in (
            self._record_mode,
            self._record_resolution,
            self._record_fps,
            self._record_download,
        ):
            handle.disabled = disabled

    def _new_video_path(self) -> Path:
        self._video_dir.mkdir(parents=True, exist_ok=True)
        return self._video_dir / f"viser-{datetime.now():%Y%m%d-%H%M%S}.mp4"

    def _start_recording(self, client: viser.ClientHandle | None) -> None:
        if client is None:
            self._set_record_status(
                '<span style="color:#e74c3c;">No client attached to the click; '
                "recording needs a connected browser.</span>"
            )
            return

        height, width = RESOLUTIONS[self._record_resolution.value]
        fps = float(self._record_fps.value)
        path = self._new_video_path()
        self._record_path = path
        self._record_stop.clear()
        self._record_stop_client = None

        if self._record_mode.value == LOCKSTEP and self._step_dt > 0.0:
            sample = self._capture_sample(0.0)
            self._lockstep = _Lockstep(
                client=client,
                path=path,
                height=height,
                width=width,
                fps=fps,
                prev=sample,
                next=sample,
                last_step_count=self._step_count,
            )
        else:
            if self._record_mode.value == LOCKSTEP:
                print(
                    "[WARN]: No step_dt on the env; falling back to realtime capture."
                )
            # Assume real-time playback until the loop measures otherwise.
            self._record_scene_boost = fps > self._base_scene_hz
            self._record_thread = threading.Thread(
                target=self._record_loop,
                args=(client, path, height, width, fps),
                name="viser-recorder",
                daemon=True,
            )
            self._record_thread.start()

        self._record_button.label = "Stop Recording"
        self._record_button.icon = viser.Icon.PLAYER_STOP
        self._set_controls_disabled(True)
        print(f"[INFO]: Recording viser view to {path}")

    def _stop_recording(
        self,
        client: viser.ClientHandle | None = None,
        error: str | None = None,
    ) -> None:
        """Stop the capture and finalize the file.

        Safe to call from the realtime capture thread itself (on a client
        error), in which case the join is skipped.
        """
        self._record_stop.set()
        self._record_scene_boost = False
        thread, self._record_thread = self._record_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_RENDER_TIMEOUT_S + 10.0)

        self._record_button.label = "Record"
        self._record_button.icon = viser.Icon.PLAYER_RECORD
        self._set_controls_disabled(False)

        path = self._record_path
        prefix = (
            f'<span style="color:#e74c3c;">Recording stopped: {error}</span><br/>'
            if error
            else ""
        )
        if path is None or not path.exists():
            self._set_record_status(prefix + "No video was written.")
            return
        size_mb = path.stat().st_size / 1e6
        print(f"[INFO]: Saved video to {path} ({size_mb:.1f} MB)")
        self._set_record_status(
            prefix + f"Saved <code>{path}</code><br/>{size_mb:.1f} MB"
        )
        if client is not None and self._record_download.value:
            self._send_download(client, path)

    def _send_download(self, client: viser.ClientHandle, path: Path) -> None:
        if path.stat().st_size > _MAX_DOWNLOAD_BYTES:
            print(f"[WARN]: {path.name} is too large to download; it stays on disk.")
            return
        try:
            client.send_file_download(path.name, path.read_bytes())
        except Exception as exc:  # noqa: BLE001 - download is best-effort.
            print(f"[WARN]: Could not send {path.name} to the browser: {exc}")

    def _capture_sample(self, sim_time: float) -> _SimSample:
        """Snapshot the render-relevant sim state onto the host."""
        data = self.env.unwrapped.sim.data
        model = self._scene.mj_model

        xpos = np.asarray(data.xpos.cpu().numpy())
        xmat = np.asarray(data.xmat.cpu().numpy())
        if xmat.shape[-1] == 9:
            xmat = xmat.reshape(*xmat.shape[:-1], 3, 3)

        if model.nmocap > 0:
            mocap_pos = np.asarray(data.mocap_pos.cpu().numpy())
            mocap_quat = np.asarray(data.mocap_quat.cpu().numpy())
        else:
            mocap_pos = mocap_quat = None

        # qpos/qvel/ctrl only feed the contact and tendon decor; skip the
        # transfer when none of it is on screen.
        decor_visible = getattr(self._scene, "_any_decor_visible", None)
        if decor_visible is not None and decor_visible():
            qpos = np.asarray(data.qpos.cpu().numpy())
            qvel = np.asarray(data.qvel.cpu().numpy())
            ctrl = np.asarray(data.ctrl.cpu().numpy()) if model.nu > 0 else None
        else:
            qpos = qvel = ctrl = None

        return _SimSample(
            sim_time=sim_time,
            xpos=xpos,
            xquat=np.asarray(vtf.SO3.from_matrix(xmat).wxyz),
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
        )

    def _push_sample(self, prev: _SimSample, nxt: _SimSample, frac: float) -> None:
        """Send a pose ``frac`` of the way from ``prev`` to ``nxt`` to the browser."""
        if frac <= 0.0:
            xpos, xquat = prev.xpos, prev.xquat
            mocap_pos, mocap_quat = prev.mocap_pos, prev.mocap_quat
        elif frac >= 1.0:
            xpos, xquat = nxt.xpos, nxt.xquat
            mocap_pos, mocap_quat = nxt.mocap_pos, nxt.mocap_quat
        else:
            xpos = prev.xpos + (nxt.xpos - prev.xpos) * frac
            xquat = _slerp(prev.xquat, nxt.xquat, frac)
            if prev.mocap_pos is None or nxt.mocap_pos is None:
                mocap_pos = mocap_quat = None
            else:
                assert prev.mocap_quat is not None and nxt.mocap_quat is not None
                mocap_pos = prev.mocap_pos + (nxt.mocap_pos - prev.mocap_pos) * frac
                mocap_quat = _slerp(prev.mocap_quat, nxt.mocap_quat, frac)

        # Decor is rebuilt by mj_forward, which nothing can interpolate.
        decor = nxt if frac >= 0.5 else prev

        # Takes _sim_lock itself, so it has to stay outside the block below.
        self._queue_debug_visualizers()
        with self._sim_lock:
            with self._server.atomic():
                self._scene.update_from_arrays(
                    xpos,
                    vtf.SO3(xquat).as_matrix(),
                    mocap_pos,
                    mocap_quat,
                    qpos=decor.qpos,
                    qvel=decor.qvel,
                    ctrl=decor.ctrl,
                )
                self._server.flush()
        self._scene.needs_update = False
        # Our push stands in for the base class's scheduled update, which is
        # what would otherwise clear these.
        self._pending_update_reasons.clear()

    @override
    def tick(self) -> bool:
        ls = self._lockstep
        if ls is None:
            return super().tick()
        try:
            return self._lockstep_tick(ls)
        except Exception as exc:  # noqa: BLE001 - never kill the viewer loop.
            self._finish_lockstep(error=f"{type(exc).__name__}: {exc}")
            return False

    def _reseed_after_reset(self, ls: _Lockstep) -> None:
        """Collapse the interpolation endpoints if the sim was rewound.

        A reset rewinds ``_step_count``, leaving the two samples straddling a
        discontinuity. Cut the video there instead of sweeping across it.
        """
        if self._step_count < ls.last_step_count:
            ls.prev = ls.next = self._capture_sample(ls.next.sim_time)
        ls.last_step_count = self._step_count

    def _lockstep_tick(self, ls: _Lockstep) -> bool:
        """One video frame: advance the sim to it, draw it, capture it.

        Replaces the base tick, which instead renders on a wall-clock schedule
        and steps to whatever fits in it.
        """
        if self._record_stop.is_set():
            self._finish_lockstep()
            return False

        if self._is_paused:
            # Hand back to the base tick so a paused view keeps its frame
            # pacing instead of syncing at the run loop's sleep rate.
            produced = ViserPlayViewer.tick(self)
            self._reseed_after_reset(ls)
            return produced

        # The base class measures dt between ticks; don't let it charge the
        # render latency to the sim budget once we hand control back.
        self._last_tick_time = time.perf_counter()
        self._process_actions()
        self._reseed_after_reset(ls)

        if self._is_paused:
            # Pause arrived with this tick's actions.
            return False

        step_dt = self._step_dt
        while ls.next.sim_time < ls.sim_target - 1e-9:
            if not self._execute_step():
                return False  # _execute_step already paused and logged.
            ls.last_step_count = self._step_count
            ls.prev = ls.next
            ls.next = self._capture_sample(ls.prev.sim_time + step_dt)

        span = ls.next.sim_time - ls.prev.sim_time
        frac = (
            0.0
            if span <= 1e-12
            else min(max((ls.sim_target - ls.prev.sim_time) / span, 0.0), 1.0)
        )

        # Status readouts, plots and camera feeds; the scene push is ours.
        self.sync_env_to_viewer()
        self._push_sample(ls.prev, ls.next, frac)

        try:
            frame = ls.client.get_render(
                height=ls.height,
                width=ls.width,
                transport_format="jpeg",
                timeout=_RENDER_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 - client may drop mid-render.
            self._finish_lockstep(error=f"{type(exc).__name__}: {exc}")
            return False

        frame = np.asarray(frame)[..., :3]
        if ls.writer is None:
            ls.shape = (frame.shape[0], frame.shape[1])
            ls.writer = media.VideoWriter(ls.path, shape=ls.shape, fps=ls.fps)
            ls.writer.__enter__()
        elif frame.shape[:2] != ls.shape:
            # The browser was resized mid-recording and the encoder is locked
            # to the first frame's shape. Hold rather than lose the time.
            ls.dropped += 1
            if ls.last_frame is None:
                return False
            frame = ls.last_frame

        ls.writer.add_image(frame)
        ls.last_frame = frame
        ls.frames_written += 1
        ls.sim_target += self._time_multiplier / ls.fps
        self._stats_frames += 1

        now = time.perf_counter()
        if now - ls.last_status > 1.0:
            ls.last_status = now
            self._set_lockstep_status(ls, now)
        return True

    def _set_lockstep_status(self, ls: _Lockstep, now: float) -> None:
        assert ls.shape is not None
        video_time = ls.frames_written / ls.fps
        wall_elapsed = now - ls.wall_start
        speed = video_time / wall_elapsed if wall_elapsed > 0 else 0.0
        native = 1.0 / self._step_dt
        source = (
            f"{native:.0f}Hz sim, interpolated"
            if ls.fps > native
            else f"{native:.0f}Hz sim"
        )
        self._set_record_status(
            '<span style="color:#e74c3c;">● Recording (lockstep)</span><br/>'
            f"<strong>File:</strong> {ls.path.name}<br/>"
            f"<strong>Size:</strong> {ls.shape[1]}x{ls.shape[0]}<br/>"
            f"<strong>Video:</strong> {video_time:.1f}s "
            f"({ls.frames_written} frames)<br/>"
            f"<strong>Distinct:</strong> {ls.fps:.0f}/{ls.fps:.0f} fps "
            f"({source})<br/>"
            f"<strong>Captured over:</strong> {wall_elapsed:.1f}s ({speed:.2f}x)"
        )

    def _finish_lockstep(self, error: str | None = None) -> None:
        ls, self._lockstep = self._lockstep, None
        if ls is None:
            return
        if ls.writer is not None:
            ls.writer.close()
        if ls.dropped:
            print(f"[WARN]: Held {ls.dropped} frames across a mid-recording resize.")
        if error is not None:
            print(f"[WARN]: Recording stopped: {error}")
        client = self._record_stop_client
        self._record_stop_client = None
        self._stop_recording(client, error=error)

    # Overrides a base @staticmethod; both call sites go through ``self``.
    def _should_submit_scene_update(  # type: ignore[override]
        self,
        counter: int,
        paused: bool,
        has_pending_updates: bool,
    ) -> bool:
        """Push scene updates every tick while a recording needs the extra rate.

        The base class submits every other tick, so the browser only sees
        ``frame_rate / 2`` distinct poses per second and a 60fps capture ends up
        half duplicates. Recording at a rate that outruns it doubles the push
        rate, but only while the sim is fast enough for that to buy anything
        (see ``_record_loop``) and only while the last scene sync fit inside a
        tick, so a machine that can't afford it isn't pushed further behind.

        Lockstep pushes its own interpolated poses, so the base schedule is
        suppressed there except while paused.
        """
        if self._lockstep is not None:
            if not paused:
                return False
        elif (
            self._record_scene_boost
            and not paused
            and self._scene_update_last_ms < self.frame_time * 1000.0
        ):
            return True
        return ViserPlayViewer._should_submit_scene_update(
            counter, paused, has_pending_updates
        )

    def _make_video_clock(self):
        """Return a callable giving elapsed *video* seconds since the call.

        Video time advances with simulated time rather than wall-clock time, so
        a sim that only runs at 0.3x real time still produces a real-time mp4.
        It is divided by the viewer's speed multiplier so a deliberate 1/2x
        playback is recorded as slow motion, and it stands still while paused.

        Falls back to wall-clock timing if the env doesn't expose ``step_dt``.
        """
        step_dt = self._step_dt
        if not step_dt:
            print("[WARN]: No step_dt on the env; recording against wall-clock time.")
            start = time.perf_counter()
            return lambda: time.perf_counter() - start

        state = {"steps": self._step_count, "time": 0.0}

        def clock() -> float:
            steps = self._step_count
            # A reset rewinds the step count; treat it as zero elapsed time
            # rather than running the clock backwards.
            delta = max(steps - state["steps"], 0)
            state["steps"] = steps
            state["time"] += delta * step_dt / max(self._time_multiplier, 1e-6)
            return state["time"]

        return clock

    def _wait_until_due(self, clock, due: float, wall_start: float) -> bool:
        """Block until video time reaches ``due``. False if recording stopped."""
        while not self._record_stop.is_set():
            video_time = clock()
            if video_time >= due:
                return True
            # Estimate the wall time left from how fast video time has been
            # advancing so far (below 1x whenever the sim can't keep up), so a
            # struggling sim is polled -- and rendered -- proportionally less.
            wall_elapsed = time.perf_counter() - wall_start
            rate = video_time / wall_elapsed if wall_elapsed > 1e-3 else 1.0
            rate = min(max(rate, 0.02), 4.0)
            self._record_stop.wait(min((due - video_time) / rate, _POLL_CAP_S))
        return False

    def _record_loop(
        self,
        client: viser.ClientHandle,
        path: Path,
        height: int,
        width: int,
        fps: float,
    ) -> None:
        """Pull frames from the client until stopped, streaming them to ``path``."""
        frame_period = 1.0 / fps
        clock = self._make_video_clock()
        wall_start = time.perf_counter()
        writer: media.VideoWriter | None = None
        shape: tuple[int, int] | None = None
        frames_written = 0
        frames_captured = 0
        dropped = 0
        last_status = 0.0
        error: str | None = None

        try:
            while not self._record_stop.is_set():
                # Only ask the browser for a frame once the next one is due on
                # the video timeline; while the sim is slow or paused, we wait.
                if not self._wait_until_due(
                    clock, frames_written * frame_period, wall_start
                ):
                    break

                try:
                    frame = client.get_render(
                        height=height,
                        width=width,
                        transport_format="jpeg",
                        timeout=_RENDER_TIMEOUT_S,
                    )
                except Exception as exc:  # noqa: BLE001 - client may drop mid-render.
                    error = f"{type(exc).__name__}: {exc}"
                    break

                frame = np.asarray(frame)[..., :3]
                if writer is None:
                    shape = (frame.shape[0], frame.shape[1])
                    writer = media.VideoWriter(path, shape=shape, fps=fps)
                    writer.__enter__()
                elif frame.shape[:2] != shape:
                    # The browser was resized mid-recording; the encoder is
                    # locked to the first frame's shape, so skip this one.
                    dropped += 1
                    continue

                # Sim time can jump several frames ahead while the browser is
                # rendering; hold the frame for as long as it is on screen.
                video_time = clock()
                repeats = max(int(video_time / frame_period) + 1 - frames_written, 1)
                for _ in range(repeats):
                    writer.add_image(frame)
                frames_written += repeats
                frames_captured += 1

                now = time.perf_counter()
                if now - last_status > 1.0:
                    last_status = now
                    wall_elapsed = now - wall_start
                    speed = video_time / wall_elapsed if wall_elapsed > 0 else 0.0
                    # Distinct poses the video needs per wall-second. A sim
                    # running below real time stretches the browser's 30Hz of
                    # scene updates over more video frames, so the boost is
                    # only worth its cost once this outruns the base rate.
                    # Hysteresis keeps it from flapping around the threshold.
                    needed_hz = fps * speed
                    if needed_hz > self._base_scene_hz:
                        self._record_scene_boost = True
                    elif needed_hz < self._base_scene_hz * 0.8:
                        self._record_scene_boost = False
                    unique = frames_captured / video_time if video_time > 0 else 0.0
                    boost = " (boosted)" if self._record_scene_boost else ""
                    assert shape is not None
                    self._set_record_status(
                        '<span style="color:#e74c3c;">● Recording</span><br/>'
                        f"<strong>File:</strong> {path.name}<br/>"
                        f"<strong>Size:</strong> {shape[1]}x{shape[0]}<br/>"
                        f"<strong>Video:</strong> {video_time:.1f}s "
                        f"({frames_written} frames)<br/>"
                        f"<strong>Distinct:</strong> {unique:.0f}/{fps:.0f} fps"
                        f"{boost}<br/>"
                        f"<strong>Captured over:</strong> {wall_elapsed:.1f}s "
                        f"({speed:.2f}x)"
                    )
        finally:
            if writer is not None:
                writer.close()

        if dropped:
            print(f"[WARN]: Dropped {dropped} resized frames while recording.")
        if error is not None:
            # Ended on its own (client gone / timed out): reset the GUI here,
            # since no one clicked "Stop Recording".
            print(f"[WARN]: Recording stopped: {error}")
            self._stop_recording(error=error)

    @override
    def close(self) -> None:
        if self._lockstep is not None:
            self._finish_lockstep()
        elif self._is_recording:
            self._stop_recording()
        super().close()

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Dict, List, Optional, Sequence

import numpy as np
import viser
import viser.transforms as viser_tf

from .config import (
    ReconstructionConfig,
    build_parser as build_reconstruction_parser,
    parse_run_args,
)
from .pipeline import IncrementalReconstructor
from .types import FrameReconstruction
from .utils import iter_image_paths, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = build_reconstruction_parser()
    parser.description = "Run progressive reconstruction with an online viser viewer."
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the viser server.")
    parser.add_argument("--port", type=int, default=8080, help="Port for the viser server.")
    parser.add_argument("--point_size", type=float, default=0.0025, help="Point size used for global map rendering.")
    parser.add_argument("--query_color", type=int, nargs=3, default=(255, 140, 0), help="RGB color for current query cameras.")
    parser.add_argument("--reference_color", type=int, nargs=3, default=(50, 205, 50), help="RGB color for current reference cameras.")
    parser.add_argument("--history_camera_color", type=int, nargs=3, default=(215, 215, 215), help="RGB color for historical cameras.")
    parser.add_argument("--frustum_scale", type=float, default=0.03, help="Scale of camera frustums in the viewer.")
    parser.add_argument("--panel_width", type=str, default="large", choices=["small", "medium", "large"], help="Width of the control panel.")
    parser.add_argument("--viewer_max_bootstrap_points", type=int, default=200000, help="Maximum number of bootstrap points displayed in the viewer.")
    parser.add_argument("--viewer_max_points_per_batch", type=int, default=40000, help="Maximum number of points displayed per added batch in the viewer.")
    parser.add_argument("--sleep_after_finish_sec", type=float, default=0.5, help="Sleep interval while keeping the viewer alive after processing.")
    return parser


def _frame_ids_from_paths(image_paths: Sequence[str]) -> List[str]:
    return [Path(path).stem for path in image_paths]


def _frame_image_shape(frame: FrameReconstruction) -> tuple[int, int]:
    shape = frame.metadata.get("image_shape")
    if isinstance(shape, (tuple, list)) and len(shape) >= 2:
        return max(1, int(shape[0])), max(1, int(shape[1]))
    if frame.image is not None:
        image = np.asarray(frame.image)
        if image.ndim >= 2:
            return max(1, int(image.shape[0])), max(1, int(image.shape[1]))
    return (96, 128)


def _frame_thumbnail(frame: FrameReconstruction) -> np.ndarray:
    thumbnail = frame.metadata.pop("viewer_thumbnail", None)
    if thumbnail is None and frame.image is not None:
        thumbnail = frame.image
    if thumbnail is None:
        return np.empty((0, 0, 3), dtype=np.uint8)
    thumb_np = np.asarray(thumbnail, dtype=np.uint8)
    if thumb_np.ndim != 3 or thumb_np.shape[-1] != 3:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return thumb_np.copy()


def _collect_fused_chunks_from_frames(
    reconstructor: IncrementalReconstructor,
    frames: Sequence[FrameReconstruction],
) -> tuple[np.ndarray, np.ndarray]:
    point_sets: List[np.ndarray] = []
    color_sets: List[np.ndarray] = []
    missing_chunk = False
    for frame in frames:
        chunk_index = frame.metadata.get("global_points_chunk_index")
        if chunk_index is None:
            continue
        chunk_index = int(chunk_index)
        if chunk_index < 0 or chunk_index >= len(reconstructor.state.global_points):
            missing_chunk = True
            break
        points = reconstructor.state.global_points[chunk_index]
        colors = reconstructor.state.global_colors[chunk_index]
        if points.size == 0:
            continue
        point_sets.append(points)
        color_sets.append(colors)
    if missing_chunk:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )
    if not point_sets:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )
    return np.concatenate(point_sets, axis=0), np.concatenate(color_sets, axis=0)


def _downsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    max_points = max(0, int(max_points))
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors

    # Uniform index sampling keeps runtime small and avoids overloading the browser.
    indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
    return points[indices], colors[indices]


def _compute_camera_fov(image_shape: Sequence[int], intrinsic: np.ndarray) -> float:
    h = int(image_shape[0]) if len(image_shape) >= 1 else 96
    fy = float(intrinsic[1, 1]) if intrinsic.shape[0] >= 2 and intrinsic.shape[1] >= 2 else 1.1 * h
    fy = fy if fy > 1e-6 else 1.1 * h
    return float(2.0 * np.arctan2(h / 2.0, fy))


def _to_viser_pose(cam2world: np.ndarray):
    pose = viser_tf.SE3.from_matrix(np.asarray(cam2world, dtype=np.float64)[:3, :4])
    return pose.rotation().wxyz, pose.translation()


def _make_placeholder_image(height: int = 96, width: int = 128) -> np.ndarray:
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    image[::8, :, :] = 220
    image[:, ::8, :] = 220
    return image


@dataclass
class BatchEvent:
    batch_idx: int
    status: str
    frame_ids: List[str]
    query_image_paths: List[str]
    selected_reference_frame_ids: List[str]
    selected_reference_image_paths: List[str]
    selected_reference_scores: List[Optional[float]]
    retrieval_time_sec: Optional[float]
    inference_time_sec: Optional[float]
    alignment_time_sec: Optional[float]
    fusion_time_sec: Optional[float]
    diagnostics: Dict[str, object]
    validation: Optional[Dict[str, object]]
    formation: Optional[Dict[str, object]]
    pose_graph_events: List[Dict[str, object]]
    added_camera_poses: List[np.ndarray]
    added_camera_intrinsics: List[np.ndarray]
    added_camera_image_shapes: List[tuple[int, int]]
    added_points: Optional[np.ndarray]
    added_colors: Optional[np.ndarray]
    current_query_poses: List[np.ndarray]
    current_query_intrinsics: List[np.ndarray]
    current_query_image_shapes: List[tuple[int, int]]
    current_query_thumbnails: List[np.ndarray]
    current_reference_poses: List[np.ndarray]
    current_reference_intrinsics: List[np.ndarray]
    current_reference_image_shapes: List[tuple[int, int]]
    global_point_count_after: int
    global_frame_count_after: int


class TimelineState:
    def __init__(self) -> None:
        self.bootstrap_frame_ids: List[str] = []
        self.bootstrap_image_paths: List[str] = []
        self.bootstrap_points = np.empty((0, 3), dtype=np.float32)
        self.bootstrap_colors = np.empty((0, 3), dtype=np.uint8)
        self.bootstrap_camera_poses: List[np.ndarray] = []
        self.bootstrap_camera_intrinsics: List[np.ndarray] = []
        self.bootstrap_camera_image_shapes: List[tuple[int, int]] = []
        self.events: List[BatchEvent] = []
        self.display_batch_idx = 0
        self.latest_online_batch_idx = 0
        self.total_images = 0
        self.bootstrap_image_count = 0

    def set_bootstrap(
        self,
        *,
        frame_ids: Sequence[str],
        image_paths: Sequence[str],
        points: np.ndarray,
        colors: np.ndarray,
        camera_poses: Sequence[np.ndarray],
        camera_intrinsics: Sequence[np.ndarray],
        camera_image_shapes: Sequence[tuple[int, int]],
    ) -> None:
        self.bootstrap_frame_ids = list(frame_ids)
        self.bootstrap_image_paths = list(image_paths)
        self.bootstrap_points = np.asarray(points, dtype=np.float32)
        self.bootstrap_colors = np.asarray(colors, dtype=np.uint8)
        self.bootstrap_camera_poses = [np.asarray(pose, dtype=np.float32) for pose in camera_poses]
        self.bootstrap_camera_intrinsics = [np.asarray(item, dtype=np.float32) for item in camera_intrinsics]
        self.bootstrap_camera_image_shapes = [(max(1, int(h)), max(1, int(w))) for h, w in camera_image_shapes]
        self.bootstrap_image_count = len(self.bootstrap_frame_ids)
        self.display_batch_idx = 0
        self.latest_online_batch_idx = 0

    def append_event(self, event: BatchEvent) -> None:
        self.events.append(event)
        self.latest_online_batch_idx = max(self.latest_online_batch_idx, int(event.batch_idx))
        self.display_batch_idx = self.latest_online_batch_idx

    def get_event(self, batch_idx: int) -> Optional[BatchEvent]:
        if batch_idx <= 0:
            return None
        offset = int(batch_idx) - 1
        if offset < 0 or offset >= len(self.events):
            return None
        return self.events[offset]

    def iter_visible_added_events(self, batch_idx: int) -> List[BatchEvent]:
        return [
            event
            for event in self.events
            if event.batch_idx <= int(batch_idx) and event.status == "added"
        ]

    def get_processed_image_count(self, batch_idx: Optional[int] = None) -> int:
        target_idx = self.display_batch_idx if batch_idx is None else int(batch_idx)
        processed = int(self.bootstrap_image_count)
        for event in self.events:
            if event.batch_idx > target_idx:
                break
            processed += len(event.frame_ids)
        return processed


class OnlineReconstructionViewer:
    def __init__(self, timeline: TimelineState, args: argparse.Namespace) -> None:
        self.timeline = timeline
        self.args = args
        self.server = viser.ViserServer(host=args.host, port=args.port)

        # Display-only transform: rotate the entire Viser world together.
        # Keep reconstruction data unchanged; all point clouds and camera frustums
        # live under /world, so a parent-frame transform preserves alignment.
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_frame(
            "/world",
            show_axes=False,
            wxyz=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),  # 180 deg around X
        )
        try:
            self.server.gui.configure_theme(
                titlebar_content=None,
                control_layout="collapsible",
                control_width=args.panel_width,
            )
        except TypeError:
            self.server.gui.configure_theme(
                titlebar_content=None,
                control_layout="collapsible",
            )

        self._lock = Lock()
        self._bootstrap_points_handle = None
        self._bootstrap_camera_handles: List[object] = []
        self._batch_point_handles: Dict[int, object] = {}
        self._batch_base_colors: Dict[int, np.ndarray] = {}
        self._batch_highlight_colors: Dict[int, np.ndarray] = {}
        self._batch_camera_handles: Dict[int, List[object]] = {}
        self._query_highlight_handles: List[object] = []
        self._reference_highlight_handles: List[object] = []
        self._placeholder_image = _make_placeholder_image()
        self._run_start_time = time.perf_counter()
        self._run_end_time: Optional[float] = None
        self._shutdown_event = Event()

        self._build_gui()
        self._status_thread = Thread(target=self._status_loop, daemon=True)
        self._status_thread.start()

    def _build_gui(self) -> None:
        with self.server.gui.add_folder("Display", expand_by_default=True):
            self.show_history_cameras_handle = self.server.gui.add_checkbox(
                "Show Historical Cameras",
                initial_value=True,
            )
            self.show_highlights_handle = self.server.gui.add_checkbox(
                "Show Current Highlights",
                initial_value=True,
            )

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self.summary_markdown_handle = self.server.gui.add_markdown("Waiting for bootstrap...")

        with self.server.gui.add_folder("Batch Images", expand_by_default=True):
            self.query_caption_handle = self.server.gui.add_markdown("**Query Images**")
            self.query_image_handles = [
                self.server.gui.add_image(
                    self._placeholder_image,
                    label=f"Q{idx + 1}",
                    visible=False,
                )
                for idx in range(5)
            ]

        @self.show_history_cameras_handle.on_update
        def _(_) -> None:
            with self._lock:
                self.render_at(self.timeline.display_batch_idx)

        @self.show_highlights_handle.on_update
        def _(_) -> None:
            with self._lock:
                self.render_at(self.timeline.display_batch_idx)


    def initialize_bootstrap(self) -> None:
        with self._lock:
            bootstrap_points, bootstrap_colors = _downsample_points(
                self.timeline.bootstrap_points,
                self.timeline.bootstrap_colors,
                self.args.viewer_max_bootstrap_points,
            )
            self._bootstrap_points_handle = self.server.scene.add_point_cloud(
                "/world/bootstrap/points",
                points=bootstrap_points,
                colors=bootstrap_colors,
                point_size=self.args.point_size,
                point_shape="circle",
                visible=True,
            )
            self._bootstrap_camera_handles = self._create_camera_handles(
                prefix="/world/bootstrap/cameras",
                frame_ids=self.timeline.bootstrap_frame_ids,
                poses=self.timeline.bootstrap_camera_poses,
                intrinsics=self.timeline.bootstrap_camera_intrinsics,
                image_shapes=self.timeline.bootstrap_camera_image_shapes,
                color=tuple(int(v) for v in self.args.history_camera_color),
                scale=float(self.args.frustum_scale),
                visible=bool(self.show_history_cameras_handle.value),
            )
            self.render_at(0)

    def append_event(self, event: BatchEvent) -> None:
        with self._lock:
            if event.status == "added" and event.added_points is not None and event.added_colors is not None:
                display_points, display_colors = _downsample_points(
                    event.added_points,
                    event.added_colors,
                    min(self.args.viewer_max_points_per_batch, 40000),
                )
                self._batch_base_colors[event.batch_idx] = display_colors
                self._batch_highlight_colors[event.batch_idx] = display_colors
                self._batch_point_handles[event.batch_idx] = self.server.scene.add_point_cloud(
                    f"/world/history/batch_{event.batch_idx:04d}/points",
                    points=display_points,
                    colors=display_colors,
                    point_size=self.args.point_size,
                    point_shape="circle",
                    visible=False,
                )
                self._batch_camera_handles[event.batch_idx] = self._create_camera_handles(
                    prefix=f"/world/history/batch_{event.batch_idx:04d}/cameras",
                    frame_ids=event.frame_ids,
                    poses=event.added_camera_poses,
                    intrinsics=event.added_camera_intrinsics,
                    image_shapes=event.added_camera_image_shapes,
                    color=tuple(int(v) for v in self.args.history_camera_color),
                    scale=float(self.args.frustum_scale),
                    visible=False,
                )
            self.render_at(self.timeline.latest_online_batch_idx)

    def mark_finished(self) -> None:
        with self._lock:
            if self._run_end_time is None:
                self._run_end_time = time.perf_counter()
            self._update_summary_panel(
                batch_idx=self.timeline.display_batch_idx,
                event=self.timeline.get_event(self.timeline.display_batch_idx),
            )

    def shutdown(self) -> None:
        self._shutdown_event.set()

    def render_at(self, batch_idx: int) -> None:
        batch_idx = int(max(0, min(batch_idx, self.timeline.latest_online_batch_idx)))
        self.timeline.display_batch_idx = batch_idx

        for event in self.timeline.iter_visible_added_events(self.timeline.latest_online_batch_idx):
            point_handle = self._batch_point_handles.get(event.batch_idx)
            if point_handle is None:
                continue
            visible = event.batch_idx <= batch_idx
            point_handle.visible = visible
            point_handle.colors = (
                self._batch_highlight_colors[event.batch_idx]
                if event.batch_idx == batch_idx
                else self._batch_base_colors[event.batch_idx]
            )
            for frustum_handle in self._batch_camera_handles.get(event.batch_idx, []):
                frustum_handle.visible = visible and bool(self.show_history_cameras_handle.value)

        for frustum_handle in self._bootstrap_camera_handles:
            frustum_handle.visible = bool(self.show_history_cameras_handle.value)

        self._clear_highlights()
        event = self.timeline.get_event(batch_idx)
        if event is not None and bool(self.show_highlights_handle.value):
            self._query_highlight_handles = self._create_camera_handles(
                prefix="/world/highlight/query",
                frame_ids=event.frame_ids,
                poses=event.current_query_poses,
                intrinsics=event.current_query_intrinsics,
                image_shapes=event.current_query_image_shapes,
                color=tuple(int(v) for v in self.args.query_color),
                scale=float(self.args.frustum_scale) * 1.1,
                visible=True,
            )
            self._reference_highlight_handles = self._create_camera_handles(
                prefix="/world/highlight/reference",
                frame_ids=event.selected_reference_frame_ids,
                poses=event.current_reference_poses,
                intrinsics=event.current_reference_intrinsics,
                image_shapes=event.current_reference_image_shapes,
                color=tuple(int(v) for v in self.args.reference_color),
                scale=float(self.args.frustum_scale) * 1.1,
                visible=True,
            )

        self._update_summary_panel(batch_idx=batch_idx, event=event)
        self._update_image_panel(event)

    def _create_camera_handles(
        self,
        *,
        prefix: str,
        frame_ids: Sequence[str],
        poses: Sequence[np.ndarray],
        intrinsics: Sequence[np.ndarray],
        image_shapes: Sequence[tuple[int, int]],
        color: tuple[int, int, int],
        scale: float,
        visible: bool,
    ) -> List[object]:
        handles: List[object] = []
        for idx, (frame_id, pose, intrinsic, image_shape) in enumerate(zip(frame_ids, poses, intrinsics, image_shapes)):
            wxyz, position = _to_viser_pose(np.asarray(pose, dtype=np.float32))
            height, width = max(1, int(image_shape[0])), max(1, int(image_shape[1]))
            frustum = self.server.scene.add_camera_frustum(
                f"{prefix}/{frame_id}_{idx}",
                wxyz=wxyz,
                position=position,
                fov=_compute_camera_fov((height, width), np.asarray(intrinsic)),
                aspect=float(width) / float(max(1, height)),
                scale=float(scale),
                line_width=2.0,
                color=color,
                visible=visible,
            )
            handles.append(frustum)
        return handles

    def _clear_highlights(self) -> None:
        for handle in self._query_highlight_handles:
            handle.remove()
        for handle in self._reference_highlight_handles:
            handle.remove()
        self._query_highlight_handles = []
        self._reference_highlight_handles = []

    def _update_summary_panel(self, *, batch_idx: int, event: Optional[BatchEvent]) -> None:
        processed_images = self.timeline.get_processed_image_count(batch_idx)
        total_images = max(0, int(self.timeline.total_images))
        elapsed_sec = self._get_elapsed_sec()
        mean_time_per_frame = elapsed_sec / processed_images if processed_images > 0 else None
        run_status_line = (
            "- Run status: finished\n"
            if self._run_end_time is not None
            else "- Run status: running\n"
        )

        if event is None:
            self.summary_markdown_handle.content = (
                "**Stage:** bootstrap\n\n"
                f"- Processed images: {processed_images}/{total_images}\n"
                f"- Frame count: {len(self.timeline.bootstrap_frame_ids)}\n"
                f"- Point count: {int(self.timeline.bootstrap_points.shape[0])}\n"
                f"- Total time: {self._fmt_duration(elapsed_sec)}\n"
                f"- Mean time / frame: {self._fmt_time_per_frame(mean_time_per_frame)}\n"
                f"{run_status_line}"
            )
            return

        lines = [
            f"**Processed images:** {processed_images}/{total_images}",
            "",
            f"- Frame count: {event.global_frame_count_after}",
            f"- Point count: {event.global_point_count_after}",
            f"- Total time: {self._fmt_duration(elapsed_sec)}",
            f"- Mean time / frame: {self._fmt_time_per_frame(mean_time_per_frame)}",
            run_status_line.rstrip(),
            f"- Current batch: {event.batch_idx}",
            f"- Batch status: `{event.status}`",
        ]
        if any(item.get("event") == "pose_graph_merged" for item in event.pose_graph_events):
            lines.append("- PGO: state updated; historical viewer handles not refreshed")
        self.summary_markdown_handle.content = "\n".join(lines)

    def _get_elapsed_sec(self) -> float:
        end_time = self._run_end_time if self._run_end_time is not None else time.perf_counter()
        return max(0.0, end_time - self._run_start_time)

    def _update_image_panel(self, event: Optional[BatchEvent]) -> None:
        if event is None:
            self.query_caption_handle.content = "**Query Images**"
            for handle in self.query_image_handles:
                handle.visible = False
            return

        self.query_caption_handle.content = f"**Query Images ({len(event.query_image_paths)})**"
        self._assign_images(
            handles=self.query_image_handles,
            thumbnails=event.current_query_thumbnails,
            labels=event.frame_ids,
        )

    def _assign_images(
        self,
        *,
        handles: Sequence[object],
        thumbnails: Sequence[np.ndarray],
        labels: Sequence[str],
    ) -> None:
        for idx, handle in enumerate(handles):
            if idx < len(thumbnails):
                thumbnail = np.asarray(thumbnails[idx], dtype=np.uint8)
                if thumbnail.ndim != 3 or thumbnail.shape[-1] != 3 or thumbnail.size == 0:
                    thumbnail = self._placeholder_image
                handle.image = thumbnail
                handle.label = labels[idx]
                handle.visible = True
            else:
                handle.visible = False

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60.0)
        remain = seconds - 60.0 * minutes
        if minutes > 0:
            return f"{minutes}m {remain:04.1f}s"
        return f"{remain:.1f}s"

    @staticmethod
    def _fmt_time_per_frame(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        if not np.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.3f}s"

    def _status_loop(self) -> None:
        while not self._shutdown_event.is_set():
            time.sleep(0.25)
            with self._lock:
                self._update_summary_panel(
                    batch_idx=self.timeline.display_batch_idx,
                    event=self.timeline.get_event(self.timeline.display_batch_idx),
                )


def _build_event_from_log(
    *,
    batch_idx: int,
    log: Dict[str, object],
    reconstructor: IncrementalReconstructor,
    image_path_lookup: Dict[str, str],
) -> BatchEvent:
    frame_ids = [str(item) for item in log.get("frame_ids", [])]
    query_image_paths = [image_path_lookup[frame_id] for frame_id in frame_ids if frame_id in image_path_lookup]

    selected_reference_frame_ids = [str(item) for item in log.get("neighbors", [])]
    if not selected_reference_frame_ids:
        selected_reference_frame_ids = [
            str(item[0]) for item in log.get("retrieved_neighbors", [])[:5]
        ]

    available_reference_frame_ids = [
        frame_id for frame_id in selected_reference_frame_ids if reconstructor.state.has_frame(frame_id)
    ]
    reference_frames = reconstructor.state.get_frames(available_reference_frame_ids) if available_reference_frame_ids else []
    selected_reference_image_paths = [frame.image_path for frame in reference_frames]

    score_lookup: Dict[str, Optional[float]] = {}
    for entry in log.get("retrieved_neighbors", []):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            try:
                score_lookup[str(entry[0])] = float(entry[1])
            except Exception:
                score_lookup[str(entry[0])] = None
    selected_reference_scores = [score_lookup.get(frame_id) for frame_id in available_reference_frame_ids]

    added_frames: List[FrameReconstruction] = []
    added_points = None
    added_colors = None
    added_camera_poses: List[np.ndarray] = []
    added_camera_intrinsics: List[np.ndarray] = []
    added_camera_image_shapes: List[tuple[int, int]] = []
    current_query_poses: List[np.ndarray] = []
    current_query_intrinsics: List[np.ndarray] = []
    current_query_image_shapes: List[tuple[int, int]] = []
    current_query_thumbnails: List[np.ndarray] = []

    if log.get("status") == "added":
        added_frames = reconstructor.state.get_frames(frame_ids)
        added_points, added_colors = _collect_fused_chunks_from_frames(reconstructor, added_frames)
        added_camera_poses = [frame.cam2world for frame in added_frames]
        added_camera_intrinsics = [frame.intrinsic for frame in added_frames]
        added_camera_image_shapes = [_frame_image_shape(frame) for frame in added_frames]
        current_query_poses = list(added_camera_poses)
        current_query_intrinsics = list(added_camera_intrinsics)
        current_query_image_shapes = list(added_camera_image_shapes)
        current_query_thumbnails = [_frame_thumbnail(frame) for frame in added_frames]

    global_point_count_after = int(sum(points.shape[0] for points in reconstructor.state.global_points))
    global_frame_count_after = int(len(reconstructor.state.frames))

    return BatchEvent(
        batch_idx=int(batch_idx),
        status=str(log.get("status", "unknown")),
        frame_ids=frame_ids,
        query_image_paths=query_image_paths,
        selected_reference_frame_ids=available_reference_frame_ids,
        selected_reference_image_paths=selected_reference_image_paths,
        selected_reference_scores=selected_reference_scores,
        retrieval_time_sec=(log.get("retrieval", {}) or {}).get("time_sec"),
        inference_time_sec=(log.get("inference", {}) or {}).get("time_sec"),
        alignment_time_sec=log.get("alignment_time_sec"),
        fusion_time_sec=log.get("fusion_time_sec"),
        diagnostics=dict(log.get("diagnostics", {}) or {}),
        validation=log.get("validation"),
        formation=log.get("formation"),
        pose_graph_events=[dict(item) for item in log.get("pose_graph_events", []) if isinstance(item, dict)],
        added_camera_poses=added_camera_poses,
        added_camera_intrinsics=added_camera_intrinsics,
        added_camera_image_shapes=added_camera_image_shapes,
        added_points=added_points,
        added_colors=added_colors,
        current_query_poses=current_query_poses,
        current_query_intrinsics=current_query_intrinsics,
        current_query_image_shapes=current_query_image_shapes,
        current_query_thumbnails=current_query_thumbnails,
        current_reference_poses=[frame.cam2world for frame in reference_frames],
        current_reference_intrinsics=[frame.intrinsic for frame in reference_frames],
        current_reference_image_shapes=[_frame_image_shape(frame) for frame in reference_frames],
        global_point_count_after=global_point_count_after,
        global_frame_count_after=global_frame_count_after,
    )


def _initialize_bootstrap_state(
    reconstructor: IncrementalReconstructor,
    timeline: TimelineState,
    image_paths: Sequence[str],
) -> None:
    init_paths = list(image_paths[: reconstructor.config.init_window])
    init_frame_ids = _frame_ids_from_paths(init_paths)
    init_frames = reconstructor.state.get_frames(init_frame_ids)
    bootstrap_points, bootstrap_colors = _collect_fused_chunks_from_frames(reconstructor, init_frames)
    timeline.set_bootstrap(
        frame_ids=init_frame_ids,
        image_paths=init_paths,
        points=bootstrap_points,
        colors=bootstrap_colors,
        camera_poses=[frame.cam2world for frame in init_frames],
        camera_intrinsics=[frame.intrinsic for frame in init_frames],
        camera_image_shapes=[_frame_image_shape(frame) for frame in init_frames],
    )


def _build_runtime_config(args: argparse.Namespace):
    from types import SimpleNamespace

    config = ReconstructionConfig(
        model_name=args.model,
        init_window=args.init_window,
        max_batch_size=args.max_batch_size,
        dynamic_batch_ref_overlap_min=args.dynamic_batch_ref_overlap_min,
        dynamic_batch_ref_overlap_ratio_min=args.dynamic_batch_ref_overlap_ratio_min,
        dynamic_batch_query_similarity_min=args.dynamic_batch_query_similarity_min,
        dynamic_batch_dominant_refs=args.dynamic_batch_dominant_refs,
        retrieval_topk=args.retrieval_topk,
        min_topk_for_alignment=args.min_topk_for_alignment,
        use_sky_mask=args.use_sky_mask,
        inference_device=args.inference_device,
        retrieval_device=args.retrieval_device,
        supscene_weights=args.supscene_weights,
        supscene_image_size=args.supscene_image_size,
        enable_profiling_sync=args.enable_profiling_sync,
        enable_detailed_profiling=args.enable_detailed_profiling,
        enable_retrieval_prefetch=not args.disable_retrieval_prefetch,
        retrieval_prefetch_count=args.retrieval_prefetch_count,
        alignment_point_conf_threshold=args.alignment_point_conf_threshold,
        alignment_grid_topk_per_cell=args.alignment_grid_topk_per_cell,
        alignment_grid_conf_min=args.alignment_grid_conf_min,
        alignment_compact_max_points=args.alignment_compact_max_points,
        alignment_compact_grid_cell_size=args.alignment_compact_grid_cell_size,
        enable_alignment_validation=not args.disable_alignment_validation,
        validation_translation_spacing_multiplier=args.validation_translation_spacing_multiplier,
        validation_spacing_estimation_samples=args.validation_spacing_estimation_samples,
        validation_spacing_estimation_max_frames=args.validation_spacing_estimation_max_frames,
        validation_median_rotation_deg_threshold=args.validation_median_rotation_deg_threshold,
        validation_max_rotation_deg_threshold=args.validation_max_rotation_deg_threshold,
        validation_scale_min=args.validation_scale_min,
        validation_scale_max=args.validation_scale_max,
        enable_register_ref_pruning_retry=args.enable_register_ref_pruning_retry,
        register_ref_pruning_candidate_z=args.register_ref_pruning_candidate_z,
        register_ref_pruning_min_similarity=args.register_ref_pruning_min_similarity,
        register_ref_pruning_max_drop_refs=args.register_ref_pruning_max_drop_refs,
        register_ref_pruning_min_refs=args.register_ref_pruning_min_refs,
        fusion_conf_threshold=args.fusion_conf_threshold,
        image_preprocess_cache_max_images=args.image_preprocess_cache_max_images,
        image_preprocess_prefetch_count=args.image_preprocess_prefetch_count,
        image_preprocess_prefetch_workers=args.image_preprocess_prefetch_workers,
        enable_pose_graph_optimization=args.enable_pose_graph_optimization,
        pose_graph_mode=args.pose_graph_mode,
        pose_opt_interval_frames=args.pose_opt_interval_frames,
        pose_opt_min_edges=args.pose_opt_min_edges,
        pose_opt_min_loop_edges=args.pose_opt_min_loop_edges,
        pose_opt_max_nfev=args.pose_opt_max_nfev,
    )
    model_args = SimpleNamespace(model=args.model, model_checkpoint=args.model_checkpoint)
    return config, model_args


def run_online_demo(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ordered_image_paths = list(iter_image_paths(args.image_dir))
    if len(ordered_image_paths) < args.init_window:
        raise ValueError(
            f"Found {len(ordered_image_paths)} images in {args.image_dir}, "
            f"but init_window={args.init_window}."
        )

    config, model_args = _build_runtime_config(args)
    reconstructor = IncrementalReconstructor(config=config, model_args=model_args)
    timeline = TimelineState()
    timeline.total_images = len(ordered_image_paths)
    viewer = OnlineReconstructionViewer(timeline=timeline, args=args)
    image_path_lookup = {Path(path).stem: path for path in ordered_image_paths}

    try:
        print(f"Starting viewer on http://{args.host}:{args.port}")
        print(f"Bootstrapping with first {args.init_window} images")
        print("Image preprocess cache: cpu")
        reconstructor.bootstrap(ordered_image_paths[: args.init_window])
        _initialize_bootstrap_state(reconstructor, timeline, ordered_image_paths)
        viewer.initialize_bootstrap()

        remaining = ordered_image_paths[args.init_window :]
        batch_idx = 1
        cursor = 0
        while cursor < len(remaining):
            batch_plan, cursor = reconstructor._form_next_dynamic_batch(
                remaining_paths=remaining,
                start_idx=cursor,
            )
            print(
                f"[Dynamic Batch] idx={batch_idx} size={len(batch_plan.frame_ids)} "
                f"frames={','.join(batch_plan.frame_ids)} reason={batch_plan.formation_reason}"
            )
            prefetch_next_paths = remaining[cursor: cursor + config.retrieval_prefetch_count]
            log = reconstructor.process_next_batch(
                batch_plan.image_paths,
                batch_idx=batch_idx,
                batch_plan=batch_plan,
                prefetch_next_paths=prefetch_next_paths,
            )
            event = _build_event_from_log(
                batch_idx=batch_idx,
                log=log,
                reconstructor=reconstructor,
                image_path_lookup=image_path_lookup,
            )
            timeline.append_event(event)
            viewer.append_event(event)
            batch_idx += 1

        reconstructor.finalize_pose_graph_optimization()
        viewer.mark_finished()
        print(f"Finished reconstruction. Total batches visualized: {timeline.latest_online_batch_idx}")
        while True:
            time.sleep(float(args.sleep_after_finish_sec))
    finally:
        viewer.shutdown()
        reconstructor.shutdown()


def main() -> None:
    args, _config_info = parse_run_args(parser=build_parser())
    run_online_demo(args)


if __name__ == "__main__":
    main()

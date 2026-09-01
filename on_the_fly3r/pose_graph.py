from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .utils import _normalize_feature_vector


@dataclass(frozen=True)
class PoseGraphEdge:
    source_id: str
    target_id: str
    relative_pose: np.ndarray
    edge_type: str = "registration"
    weight: float = 1.0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PoseGraphSnapshot:
    frame_ids: List[str]
    cam2world: np.ndarray
    edges: List[PoseGraphEdge]
    snapshot_frame_count: int
    trigger_reason: str = "interval"


@dataclass
class PoseGraphResult:
    frame_ids: List[str]
    cam2world_initial: np.ndarray
    cam2world_refined: np.ndarray
    snapshot_frame_count: int
    summary: Dict[str, object]


@dataclass(frozen=True)
class OnlinePoseGraphJob:
    snapshot: PoseGraphSnapshot
    check_index: int
    new_loop_edges: int
    interval_frames: int
    snapshot_build_time_sec: float


@dataclass
class OnlinePoseGraphOutcome:
    job: OnlinePoseGraphJob
    result: Optional[PoseGraphResult]
    effective_frame_ids: List[str]
    effective_cam2world: np.ndarray
    accepted: bool
    discard_reason: Optional[str]
    error: Optional[str]
    online_total_time_sec: float


def _run_online_pose_graph_job(
    job: OnlinePoseGraphJob,
    previous_future: Optional[Future],
    *,
    max_nfev: int,
) -> OnlinePoseGraphOutcome:
    """Run one queued PGO job without reading or writing reconstruction state."""
    online_start = time.perf_counter()
    initial = np.asarray(job.snapshot.cam2world, dtype=np.float32).copy()

    if previous_future is not None:
        previous = previous_future.result()
        previous_by_id = {
            frame_id: previous.effective_cam2world[idx]
            for idx, frame_id in enumerate(previous.effective_frame_ids)
        }
        for idx, frame_id in enumerate(job.snapshot.frame_ids):
            previous_pose = previous_by_id.get(frame_id)
            if previous_pose is not None:
                initial[idx] = previous_pose

    prepared_snapshot = PoseGraphSnapshot(
        frame_ids=list(job.snapshot.frame_ids),
        cam2world=initial,
        edges=list(job.snapshot.edges),
        snapshot_frame_count=int(job.snapshot.snapshot_frame_count),
        trigger_reason=job.snapshot.trigger_reason,
    )
    try:
        result = optimize_pose_graph(
            prepared_snapshot,
            max_nfev=int(max_nfev),
        )
    except Exception as exc:
        return OnlinePoseGraphOutcome(
            job=job,
            result=None,
            effective_frame_ids=list(prepared_snapshot.frame_ids),
            effective_cam2world=initial,
            accepted=False,
            discard_reason=None,
            error=str(exc),
            online_total_time_sec=time.perf_counter() - online_start,
        )

    initial_cost = float(result.summary.get("initial_cost") or 0.0)
    final_cost = float(result.summary.get("final_cost") or 0.0)
    success = bool(result.summary.get("success", False))
    finite_costs = bool(np.isfinite(initial_cost) and np.isfinite(final_cost))
    accepted = bool(success and finite_costs and final_cost <= initial_cost + 1e-9)
    if not success:
        discard_reason = "optimizer_unsuccessful"
    elif not finite_costs:
        discard_reason = "non_finite_cost"
    elif not accepted:
        discard_reason = "cost_increased"
    else:
        discard_reason = None

    effective = result.cam2world_refined if accepted else initial
    return OnlinePoseGraphOutcome(
        job=job,
        result=result,
        effective_frame_ids=list(result.frame_ids),
        effective_cam2world=np.asarray(effective, dtype=np.float32).copy(),
        accepted=accepted,
        discard_reason=discard_reason,
        error=None,
        online_total_time_sec=time.perf_counter() - online_start,
    )


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def optimize_pose_graph(
    snapshot: PoseGraphSnapshot,
    *,
    max_nfev: int = 50,
    huber_scale: float = 1.0,
) -> PoseGraphResult:
    """Optimize an SE3 pose graph with GTSAM.

    GTSAM is intentionally the only backend. Missing GTSAM therefore raises
    ImportError instead of silently switching to a different optimizer.
    """
    import gtsam

    start_time = time.perf_counter()
    frame_ids = list(snapshot.frame_ids)
    initial = _project_poses(np.asarray(snapshot.cam2world, dtype=np.float64))
    id_to_idx = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
    edges = _usable_edges(snapshot.edges, id_to_idx)

    if len(frame_ids) < 2 or not edges:
        summary = {
            "success": False,
            "reason": "not_enough_nodes_or_edges",
            "num_frames": int(len(frame_ids)),
            "num_edges": int(len(edges)),
            "initial_cost": 0.0,
            "final_cost": 0.0,
            "optimization_time_sec": time.perf_counter() - start_time,
            "trigger_reason": snapshot.trigger_reason,
            "backend": "gtsam",
            "model": "se3",
        }
        return PoseGraphResult(
            frame_ids=frame_ids,
            cam2world_initial=initial.astype(np.float32),
            cam2world_refined=initial.astype(np.float32),
            snapshot_frame_count=int(snapshot.snapshot_frame_count),
            summary=summary,
        )

    components = _DisjointSet(len(frame_ids))
    for edge in edges:
        components.union(id_to_idx[edge.source_id], id_to_idx[edge.target_id])
    anchors = _component_anchors(components, len(frame_ids))
    translation_scale = _estimate_translation_scale(edges)

    graph_build_start = time.perf_counter()
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    keys = [gtsam.symbol("x", idx) for idx in range(len(frame_ids))]
    for idx, key in enumerate(keys):
        values.insert(key, gtsam.Pose3(initial[idx]))

    prior_translation_sigma = max(translation_scale, 1.0) * 1e-6
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray([1e-6, 1e-6, 1e-6, prior_translation_sigma,
                    prior_translation_sigma, prior_translation_sigma], dtype=np.float64)
    )
    for anchor_idx in anchors:
        graph.add(gtsam.PriorFactorPose3(
            keys[anchor_idx],
            gtsam.Pose3(initial[anchor_idx]),
            prior_noise,
        ))

    base_sigmas = np.asarray(
        [1.0, 1.0, 1.0, translation_scale, translation_scale, translation_scale],
        dtype=np.float64,
    )
    for edge in edges:
        weight_scale = 1.0 / np.sqrt(max(float(edge.weight), 1e-6))
        noise = gtsam.noiseModel.Diagonal.Sigmas(base_sigmas * weight_scale)
        if huber_scale > 0:
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(float(huber_scale)),
                noise,
            )
        graph.add(gtsam.BetweenFactorPose3(
            keys[id_to_idx[edge.source_id]],
            keys[id_to_idx[edge.target_id]],
            gtsam.Pose3(_project_pose(edge.relative_pose)),
            noise,
        ))
    graph_build_time = time.perf_counter() - graph_build_start

    initial_cost = float(graph.error(values))
    solver_start = time.perf_counter()
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max(1, int(max_nfev)))
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    optimized_values = optimizer.optimize()
    solver_time = time.perf_counter() - solver_start
    final_cost = float(graph.error(optimized_values))

    refined = np.empty_like(initial)
    for idx, key in enumerate(keys):
        refined[idx] = _project_pose(
            np.asarray(optimized_values.atPose3(key).matrix(), dtype=np.float64)
        )

    initial_blocks = _edge_residual_blocks(initial, edges, id_to_idx, translation_scale)
    final_blocks = _edge_residual_blocks(refined, edges, id_to_idx, translation_scale)
    iterations = None
    if hasattr(optimizer, "iterations"):
        try:
            iterations = int(optimizer.iterations())
        except Exception:
            pass

    summary = {
        "success": bool(np.isfinite(final_cost)),
        "reason": "ok" if np.isfinite(final_cost) else "optimizer_failed",
        "num_frames": int(len(frame_ids)),
        "num_edges": int(len(edges)),
        "edge_type_counts": _edge_type_counts(edges),
        "num_variable_frames": int(len(frame_ids) - len(anchors)),
        "num_anchor_frames": int(len(anchors)),
        "anchor_frame_ids": [frame_ids[idx] for idx in anchors],
        "translation_scale": float(translation_scale),
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "initial_residual": _residual_stats(_flatten_residuals(initial_blocks)),
        "final_residual": _residual_stats(_flatten_residuals(final_blocks)),
        "initial_residual_by_edge_type": _residual_stats_by_edge_type(initial_blocks),
        "final_residual_by_edge_type": _residual_stats_by_edge_type(final_blocks),
        "optimization_success": bool(np.isfinite(final_cost)),
        "optimization_status": 0,
        "optimization_message": "gtsam_levenberg_marquardt",
        "optimization_nfev": int(iterations if iterations is not None else max(1, int(max_nfev))),
        "optimization_time_sec": time.perf_counter() - start_time,
        "graph_build_time_sec": graph_build_time,
        "solver_time_sec": solver_time,
        "trigger_reason": snapshot.trigger_reason,
        "backend": "gtsam",
        "model": "se3",
    }
    return PoseGraphResult(
        frame_ids=frame_ids,
        cam2world_initial=initial.astype(np.float32),
        cam2world_refined=refined.astype(np.float32),
        snapshot_frame_count=int(snapshot.snapshot_frame_count),
        summary=summary,
    )


def _usable_edges(
    edges: Sequence[PoseGraphEdge],
    id_to_idx: Dict[str, int],
) -> List[PoseGraphEdge]:
    usable = []
    for edge in edges:
        if edge.source_id not in id_to_idx or edge.target_id not in id_to_idx:
            continue
        if edge.source_id == edge.target_id:
            continue
        relative = np.asarray(edge.relative_pose, dtype=np.float64)
        if relative.shape != (4, 4) or not np.isfinite(relative).all():
            continue
        usable.append(edge)
    return usable


def _component_anchors(components: _DisjointSet, size: int) -> List[int]:
    anchors: Dict[int, int] = {}
    for idx in range(size):
        root = components.find(idx)
        anchors[root] = min(idx, anchors.get(root, idx))
    return sorted(anchors.values())


def _estimate_translation_scale(edges: Sequence[PoseGraphEdge]) -> float:
    lengths = [
        float(np.linalg.norm(np.asarray(edge.relative_pose, dtype=np.float64)[:3, 3]))
        for edge in edges
    ]
    valid = [length for length in lengths if np.isfinite(length) and length > 1e-8]
    return max(float(np.median(valid)), 1e-3) if valid else 1.0


def _edge_residual_blocks(
    poses: np.ndarray,
    edges: Sequence[PoseGraphEdge],
    id_to_idx: Dict[str, int],
    translation_scale: float,
) -> List[Dict[str, object]]:
    blocks = []
    for edge in edges:
        source = poses[id_to_idx[edge.source_id]]
        target = poses[id_to_idx[edge.target_id]]
        predicted = np.linalg.inv(source) @ target
        measured = _project_pose(edge.relative_pose)
        error = np.linalg.inv(measured) @ predicted
        weight = np.sqrt(max(float(edge.weight), 1e-6))
        residual = np.concatenate([
            _rotation_log_vector(error[:3, :3]),
            error[:3, 3] / max(float(translation_scale), 1e-8),
        ]) * weight
        blocks.append({"edge_type": edge.edge_type, "residual": residual})
    return blocks


def _rotation_log_vector(rotation: np.ndarray) -> np.ndarray:
    rotation = _project_rotation(rotation)
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    skew = np.asarray([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ], dtype=np.float64)
    if angle < 1e-8:
        return 0.5 * skew
    sin_angle = float(np.sin(angle))
    if abs(sin_angle) > 1e-6:
        return (0.5 * angle / sin_angle) * skew
    eigenvalues, eigenvectors = np.linalg.eig(rotation)
    axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    return angle * axis


def _flatten_residuals(blocks: Sequence[Dict[str, object]]) -> np.ndarray:
    if not blocks:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate([np.asarray(block["residual"], dtype=np.float64) for block in blocks])


def _residual_stats(residuals: np.ndarray) -> Dict[str, float]:
    residuals = np.abs(np.asarray(residuals, dtype=np.float64))
    if residuals.size == 0:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(residuals)),
        "median": float(np.median(residuals)),
        "max": float(np.max(residuals)),
    }


def _residual_stats_by_edge_type(
    blocks: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[np.ndarray]] = {}
    for block in blocks:
        grouped.setdefault(str(block["edge_type"]), []).append(
            np.asarray(block["residual"], dtype=np.float64)
        )
    return {
        edge_type: _residual_stats(np.concatenate(values))
        for edge_type, values in grouped.items()
    }


def _edge_type_counts(edges: Sequence[PoseGraphEdge]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for edge in edges:
        counts[edge.edge_type] = int(counts.get(edge.edge_type, 0)) + 1
    return counts


def _project_rotation(rotation: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    projected = left @ right_t
    if np.linalg.det(projected) < 0:
        left[:, -1] *= -1.0
        projected = left @ right_t
    return projected


def _project_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    projected = np.eye(4, dtype=np.float64)
    projected[:3, :3] = _project_rotation(pose[:3, :3])
    projected[:3, 3] = pose[:3, 3]
    return projected


def _project_poses(poses: np.ndarray) -> np.ndarray:
    return np.asarray([_project_pose(pose) for pose in poses], dtype=np.float64)


class PoseGraphMixin:
    def _record_pose_graph_observations(
        self,
        *,
        subset: SubsetInferenceResult,
        neighbor_frames: List[FrameReconstruction],
        fresh_frame_ids: Sequence[str],
        alignment: AlignmentResult,
        scored_neighbors_per_frame: Dict[str, List[tuple[str, float]]],
    ) -> None:
        if (
            not self.config.enable_pose_graph_optimization
        ):
            return

        start_count = len(self.pose_graph_edges)
        new_loop_count = 0
        num_refs = len(neighbor_frames)
        if num_refs <= 0:
            return

        aligned_poses = [
            self._transform_pose(alignment.transform_local_to_global, pose)
            for pose in subset.cam2world
        ]
        id_to_local = {frame_id: idx for idx, frame_id in enumerate(subset.frame_ids)}
        alignment_weight = self._pose_graph_alignment_weight(alignment.diagnostics)

        for ref_idx, ref_frame in enumerate(neighbor_frames):
            ref_order = self.state.get_frame_order_index(ref_frame.frame_id)
            for fresh_id in fresh_frame_ids:
                fresh_local_idx = id_to_local.get(fresh_id)
                if fresh_local_idx is None:
                    continue
                fresh_order = self.state.get_frame_order_index(fresh_id)
                edge_type = "registration"
                if ref_order is not None and fresh_order is not None:
                    if abs(int(fresh_order) - int(ref_order)) >= int(self.config.pose_graph_loop_min_separation):
                        edge_type = "loop"
                        new_loop_count += 1
                self.pose_graph_edges.append(
                    PoseGraphEdge(
                        source_id=ref_frame.frame_id,
                        target_id=fresh_id,
                        relative_pose=self._relative_pose(aligned_poses[ref_idx], aligned_poses[fresh_local_idx]),
                        edge_type=edge_type,
                        weight=alignment_weight,
                        metadata={
                            "mode": alignment.diagnostics.get("mode"),
                            "measurement_source": "aligned_subset",
                        },
                    )
                )

        fresh_local_indices = [id_to_local[frame_id] for frame_id in fresh_frame_ids if frame_id in id_to_local]
        for left_pos, left_local_idx in enumerate(fresh_local_indices):
            for right_local_idx in fresh_local_indices[left_pos + 1:]:
                self.pose_graph_edges.append(
                    PoseGraphEdge(
                        source_id=subset.frame_ids[left_local_idx],
                        target_id=subset.frame_ids[right_local_idx],
                        relative_pose=self._relative_pose(aligned_poses[left_local_idx], aligned_poses[right_local_idx]),
                        edge_type="intra_batch",
                        weight=alignment_weight,
                        metadata={},
                    )
                )

        self._record_local_continuity_edges(
            fresh_frame_ids=fresh_frame_ids,
            scored_neighbors_per_frame=scored_neighbors_per_frame,
        )

        added_count = len(self.pose_graph_edges) - start_count
        if added_count > 0:
            event = {
                "event": "edges_recorded",
                "added_edges": int(added_count),
                "total_edges": int(len(self.pose_graph_edges)),
                "frame_count": int(self.state.frame_count()),
            }
            self.pose_graph_events.append(event)
        self._online_pgo_new_loop_edges_since_check += int(new_loop_count)

    def _record_local_continuity_edges(
        self,
        *,
        fresh_frame_ids: Sequence[str],
        scored_neighbors_per_frame: Dict[str, List[tuple[str, float]]],
    ) -> None:
        order = self.state.ordered_frame_ids()
        if len(order) < 2:
            return
        order_index = {frame_id: idx for idx, frame_id in enumerate(order)}
        for frame_id in fresh_frame_ids:
            idx = order_index.get(frame_id)
            if idx is None or idx <= 0:
                continue
            prev_id = order[idx - 1]
            if not self._accept_local_continuity_edge(
                prev_id=prev_id,
                frame_id=frame_id,
                scored_neighbors_per_frame=scored_neighbors_per_frame,
            ):
                continue
            prev_frame = self.state.get_frame(prev_id)
            frame = self.state.get_frame(frame_id)
            translation_step = float(np.linalg.norm(frame.cam2world[:3, 3] - prev_frame.cam2world[:3, 3]))
            rotation_step = self._rotation_angle_deg(prev_frame.cam2world[:3, :3], frame.cam2world[:3, :3])
            self._continuity_translation_steps.append(translation_step)
            self._continuity_rotation_steps_deg.append(rotation_step)
            self._continuity_translation_steps = self._continuity_translation_steps[-200:]
            self._continuity_rotation_steps_deg = self._continuity_rotation_steps_deg[-200:]
            self.pose_graph_edges.append(
                PoseGraphEdge(
                    source_id=prev_id,
                    target_id=frame_id,
                    relative_pose=self._relative_pose(prev_frame.cam2world, frame.cam2world),
                    edge_type="local_continuity",
                    weight=1.0,
                    metadata={
                        "translation_step": translation_step,
                        "rotation_step_deg": rotation_step,
                    },
                )
            )

    def _accept_local_continuity_edge(
        self,
        *,
        prev_id: str,
        frame_id: str,
        scored_neighbors_per_frame: Dict[str, List[tuple[str, float]]],
    ) -> bool:
        prev_frame = self.state.get_frame(prev_id)
        frame = self.state.get_frame(frame_id)
        visual_ok = False
        if prev_frame.retrieval_vector is not None and frame.retrieval_vector is not None:
            sim = float(np.dot(
                _normalize_feature_vector(prev_frame.retrieval_vector),
                _normalize_feature_vector(frame.retrieval_vector),
            ))
            visual_ok = sim >= float(self.config.pose_graph_continuity_similarity_min)

        prev_neighbors = {
            str(item[0])
            for item in prev_frame.metadata.get("scored_neighbors", [])
            if isinstance(item, (tuple, list)) and item
        }
        current_neighbors = {
            str(item[0])
            for item in scored_neighbors_per_frame.get(frame_id, [])
            if isinstance(item, (tuple, list)) and item
        }
        overlap_ok = (
            len(prev_neighbors & current_neighbors)
            >= int(self.config.pose_graph_continuity_ref_overlap_min)
        )
        if not visual_ok and not overlap_ok:
            return False

        translation_step = float(np.linalg.norm(frame.cam2world[:3, 3] - prev_frame.cam2world[:3, 3]))
        rotation_step = self._rotation_angle_deg(prev_frame.cam2world[:3, :3], frame.cam2world[:3, :3])
        if rotation_step > float(self.config.pose_graph_continuity_rotation_max_deg):
            return False
        if self._continuity_translation_steps:
            median_step = float(np.median(np.asarray(self._continuity_translation_steps, dtype=np.float64)))
            threshold = max(1e-6, median_step * float(self.config.pose_graph_continuity_step_multiplier))
            if translation_step > threshold:
                return False
        return True

    @staticmethod
    def _pose_graph_alignment_weight(diagnostics: Dict[str, object]) -> float:
        num_corr = float(diagnostics.get("num_correspondences") or 0.0)
        num_inliers = float(diagnostics.get("num_inliers") or 0.0)
        median_residual = float(diagnostics.get("median_point_residual") or 0.0)
        inlier_ratio = num_inliers / num_corr if num_corr > 0 else 1.0
        corr_scale = min(3.0, max(0.5, np.sqrt(max(num_corr, 1.0) / 1000.0)))
        residual_scale = 1.0 / (1.0 + max(0.0, median_residual))
        return float(np.clip(corr_scale * max(inlier_ratio, 0.1) * residual_scale, 0.1, 5.0))

    @staticmethod
    def _relative_pose(source_cam2world: np.ndarray, target_cam2world: np.ndarray) -> np.ndarray:
        source = np.asarray(source_cam2world, dtype=np.float64)
        target = np.asarray(target_cam2world, dtype=np.float64)
        return (np.linalg.inv(source) @ target).astype(np.float32)

    def _build_pose_graph_snapshot(
        self,
        trigger_reason: str,
    ) -> Optional[PoseGraphSnapshot]:
        frame_ids = self.state.ordered_frame_ids()
        if len(frame_ids) < 2:
            return None
        valid_ids = set(frame_ids)
        edges = [
            PoseGraphEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                relative_pose=np.asarray(edge.relative_pose, dtype=np.float32).copy(),
                edge_type=edge.edge_type,
                weight=float(edge.weight),
                metadata=dict(edge.metadata),
            )
            for edge in self.pose_graph_edges
            if (
                edge.source_id in valid_ids
                and edge.target_id in valid_ids
            )
        ]
        if len(edges) < int(self.config.pose_opt_min_edges):
            return None
        poses = np.stack(
            [self.state.get_optimized_pose(frame_id) for frame_id in frame_ids],
            axis=0,
        ).astype(np.float32)
        return PoseGraphSnapshot(
            frame_ids=list(frame_ids),
            cam2world=poses,
            edges=edges,
            snapshot_frame_count=len(frame_ids),
            trigger_reason=trigger_reason,
        )

    def _merge_pose_graph_result(self, result: PoseGraphResult) -> Dict[str, object]:
        merge_start = time.perf_counter()
        public_summary = self._public_pose_graph_summary(result.summary)
        if not bool(result.summary.get("success", False)):
            event = {
                "event": "pose_graph_discarded",
                "reason": "optimizer_unsuccessful",
                "snapshot_frame_count": int(result.snapshot_frame_count),
                "current_frame_count": int(self.state.frame_count()),
                "summary": public_summary,
            }
            self._print_pose_graph_discard(event)
            return event
        current_ids = self.state.ordered_frame_ids()
        snapshot_count = int(result.snapshot_frame_count)
        if snapshot_count > len(current_ids):
            event = {
                "event": "pose_graph_discarded",
                "reason": "frame_order_mismatch",
                "snapshot_frame_count": snapshot_count,
                "current_frame_count": int(len(current_ids)),
                "summary": public_summary,
            }
            self._print_pose_graph_discard(event)
            return event
        snapshot_known_ids = set(current_ids[:snapshot_count])
        missing_snapshot_ids = [frame_id for frame_id in result.frame_ids if frame_id not in snapshot_known_ids]
        if missing_snapshot_ids:
            event = {
                "event": "pose_graph_discarded",
                "reason": "snapshot_frame_id_mismatch",
                "snapshot_frame_count": snapshot_count,
                "current_frame_count": int(len(current_ids)),
                "missing_frame_ids": missing_snapshot_ids[:5],
                "summary": public_summary,
            }
            self._print_pose_graph_discard(event)
            return event

        refined_by_id = {
            frame_id: np.asarray(result.cam2world_refined[idx], dtype=np.float64)
            for idx, frame_id in enumerate(result.frame_ids)
        }
        initial_by_id = {
            frame_id: np.asarray(result.cam2world_initial[idx], dtype=np.float64)
            for idx, frame_id in enumerate(result.frame_ids)
        }
        translation_deltas: List[float] = []
        rotation_deltas_deg: List[float] = []
        optimized_count = 0
        updated_count = 0
        for frame_id in result.frame_ids:
            frame = self.state.frames.get(frame_id)
            if frame is None:
                continue
            old_pose = initial_by_id[frame_id]
            new_pose = refined_by_id[frame_id]
            delta = new_pose @ np.linalg.inv(old_pose)
            translation_deltas.append(float(np.linalg.norm(delta[:3, 3])))
            rotation_deltas_deg.append(self._delta_rotation_deg(delta))
            if not self._is_near_identity_delta(delta):
                self._apply_optimized_pose_update(frame_id=frame_id, new_pose=new_pose)
                updated_count += 1
            optimized_count += 1
        event = {
            "event": "pose_graph_merged",
            "optimized_frame_count": int(optimized_count),
            "updated_frame_count": int(updated_count),
            "snapshot_frame_count": int(snapshot_count),
            "num_snapshot_frames": int(len(result.frame_ids)),
            "current_frame_count": int(self.state.frame_count()),
            "initial_cost": result.summary.get("initial_cost"),
            "final_cost": result.summary.get("final_cost"),
            "pgo_time_sec": result.summary.get("optimization_time_sec"),
            "pose_update_time_sec": time.perf_counter() - merge_start,
            "pose_delta_stats": {
                "translation": self._value_stats(translation_deltas),
                "rotation_deg": self._value_stats(rotation_deltas_deg),
            },
            "summary": public_summary,
        }
        self._last_pose_graph_debug_payload = self._make_pose_graph_debug_payload(result)
        initial_cost = float(event["initial_cost"] or 0.0)
        final_cost = float(event["final_cost"] or 0.0)
        delta_translation_median = float(event["pose_delta_stats"]["translation"]["median"])
        delta_rotation_median = float(event["pose_delta_stats"]["rotation_deg"]["median"])
        print(
            "[Pose Graph Merged] "
            f"optimized={optimized_count} updated={updated_count} "
            f"cost={initial_cost:.6f}->{final_cost:.6f} "
            f"pgo={float(event['pgo_time_sec'] or 0.0):.2f}s "
            f"pose_update={event['pose_update_time_sec']:.2f}s "
            f"delta_t_med={delta_translation_median:.6f} "
            f"delta_r_med={delta_rotation_median:.3f}deg"
        )
        return event

    def maybe_run_online_pose_graph_optimization(
        self,
        accepted_frame_count: int,
    ) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        if (
            not self.config.enable_pose_graph_optimization
            or self.config.pose_graph_mode not in {"online", "online_and_final"}
        ):
            return events

        self._online_pgo_accepted_frames_since_check += max(0, int(accepted_frame_count))
        interval = int(self.config.pose_opt_interval_frames)
        if self._online_pgo_accepted_frames_since_check < interval:
            return events

        self._online_pgo_accepted_frames_since_check -= interval
        self._online_pgo_check_count += 1
        new_loop_edges = int(self._online_pgo_new_loop_edges_since_check)
        self._online_pgo_new_loop_edges_since_check = 0

        if new_loop_edges < int(self.config.pose_opt_min_loop_edges):
            event = {
                "event": "pose_graph_online_skipped",
                "phase": "online",
                "reason": "no_new_loop_edges",
                "check_index": int(self._online_pgo_check_count),
                "frame_count": int(self.state.frame_count()),
                "new_loop_edges": int(new_loop_edges),
                "interval_frames": int(interval),
            }
            self.pose_graph_events.append(event)
            events.append(event)
            return events

        build_start = time.perf_counter()
        snapshot = self._build_pose_graph_snapshot("online_interval_loop")
        snapshot_build_time = time.perf_counter() - build_start
        if snapshot is None:
            event = {
                "event": "pose_graph_online_skipped",
                "phase": "online",
                "reason": "not_enough_edges",
                "check_index": int(self._online_pgo_check_count),
                "frame_count": int(self.state.frame_count()),
                "num_edges": int(len(self.pose_graph_edges)),
                "min_edges": int(self.config.pose_opt_min_edges),
                "new_loop_edges": int(new_loop_edges),
                "interval_frames": int(interval),
                "snapshot_build_time_sec": snapshot_build_time,
            }
            self.pose_graph_events.append(event)
            events.append(event)
            return events

        executor = self._ensure_online_pose_graph_executor()
        job = OnlinePoseGraphJob(
            snapshot=snapshot,
            check_index=int(self._online_pgo_check_count),
            new_loop_edges=int(new_loop_edges),
            interval_frames=int(interval),
            snapshot_build_time_sec=float(snapshot_build_time),
        )
        previous_future = getattr(self, "_online_pgo_last_future", None)
        future = executor.submit(
            _run_online_pose_graph_job,
            job,
            previous_future,
            max_nfev=int(self.config.pose_opt_max_nfev),
        )
        self._online_pgo_last_future = future
        self._online_pgo_pending_jobs.append((job, future))
        event = {
            "event": "pose_graph_online_scheduled",
            "phase": "online",
            "check_index": int(job.check_index),
            "frame_count": int(self.state.frame_count()),
            "num_snapshot_frames": int(len(snapshot.frame_ids)),
            "num_edges": int(len(snapshot.edges)),
            "new_loop_edges": int(new_loop_edges),
            "interval_frames": int(interval),
            "snapshot_build_time_sec": snapshot_build_time,
        }
        print(
            "[Pose Graph Online Scheduled] "
            f"check={job.check_index} frames={len(snapshot.frame_ids)} "
            f"edges={len(snapshot.edges)} new_loops={new_loop_edges} "
            f"max_iter={self.config.pose_opt_max_nfev}"
        )
        self.pose_graph_events.append(event)
        events.append(event)
        return events

    def _ensure_online_pose_graph_executor(self) -> ThreadPoolExecutor:
        executor = getattr(self, "_online_pgo_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="online-pgo")
            self._online_pgo_executor = executor
            self._online_pgo_pending_jobs = []
            self._online_pgo_last_future = None
        return executor

    def _consume_online_pose_graph_outcome(
        self,
        outcome: OnlinePoseGraphOutcome,
    ) -> Dict[str, object]:
        job = outcome.job
        if outcome.error is not None:
            event = {
                "event": "pose_graph_failed",
                "phase": "online",
                "error": outcome.error,
                "check_index": int(job.check_index),
                "frame_count": int(self.state.frame_count()),
                "num_snapshot_frames": int(len(job.snapshot.frame_ids)),
                "num_edges": int(len(job.snapshot.edges)),
                "new_loop_edges": int(job.new_loop_edges),
            }
        elif not outcome.accepted:
            result = outcome.result
            event = {
                "event": "pose_graph_discarded",
                "phase": "online",
                "reason": outcome.discard_reason,
                "check_index": int(job.check_index),
                "snapshot_frame_count": int(job.snapshot.snapshot_frame_count),
                "current_frame_count": int(self.state.frame_count()),
                "new_loop_edges": int(job.new_loop_edges),
                "summary": (
                    self._public_pose_graph_summary(result.summary)
                    if result is not None
                    else {}
                ),
            }
            self._print_pose_graph_discard(event)
        else:
            if outcome.result is None:
                raise RuntimeError("Accepted online PGO outcome has no optimization result.")
            event = self._merge_pose_graph_result(outcome.result)
            event["phase"] = "online"
            event["check_index"] = int(job.check_index)
            event["new_loop_edges"] = int(job.new_loop_edges)
            if event.get("event") == "pose_graph_merged":
                self._online_pgo_run_count += 1

        event["interval_frames"] = int(job.interval_frames)
        event["snapshot_build_time_sec"] = float(job.snapshot_build_time_sec)
        event["online_total_time_sec"] = float(outcome.online_total_time_sec)
        print(
            "[Pose Graph Online Finished] "
            f"check={job.check_index} event={event.get('event')} "
            f"frames={event.get('num_snapshot_frames', len(job.snapshot.frame_ids))} "
            f"edges={len(job.snapshot.edges)} new_loops={job.new_loop_edges} "
            f"total={outcome.online_total_time_sec:.2f}s"
        )
        self.pose_graph_events.append(event)
        return event

    def _drain_online_pose_graph_jobs(self) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        pending_jobs = list(getattr(self, "_online_pgo_pending_jobs", []))
        for _job, future in pending_jobs:
            outcome = future.result()
            events.append(self._consume_online_pose_graph_outcome(outcome))
        self._online_pgo_pending_jobs = []
        self._online_pgo_last_future = None
        return events

    def shutdown_online_pose_graph_worker(self, *, wait: bool = True) -> None:
        executor = getattr(self, "_online_pgo_executor", None)
        if executor is None:
            return
        executor.shutdown(wait=wait, cancel_futures=False)
        self._online_pgo_executor = None

    def finalize_pose_graph_optimization(self) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        if not self.config.enable_pose_graph_optimization:
            return events
        finalize_start = time.perf_counter()
        events.extend(self._drain_online_pose_graph_jobs())
        self.shutdown_online_pose_graph_worker(wait=True)
        if self.config.pose_graph_mode == "online":
            self.runtime_stats["pose_graph_finalize_time_sec"] = time.perf_counter() - finalize_start
            return events

        build_start = time.perf_counter()
        snapshot = self._build_pose_graph_snapshot("final")
        snapshot_build_time = time.perf_counter() - build_start
        if snapshot is None:
            event = {
                "event": "pose_graph_final_skipped",
                "phase": "final",
                "reason": "not_enough_edges",
                "frame_count": int(self.state.frame_count()),
                "num_edges": int(len(self.pose_graph_edges)),
                "min_edges": int(self.config.pose_opt_min_edges),
                "snapshot_build_time_sec": snapshot_build_time,
            }
            print(
                "[Pose Graph Final Skipped] "
                f"reason={event['reason']} frames={event['frame_count']} "
                f"edges={event['num_edges']} min_edges={event['min_edges']}"
            )
            self.pose_graph_events.append(event)
            events.append(event)
            self.runtime_stats["pose_graph_finalize_time_sec"] = time.perf_counter() - finalize_start
            return events

        snapshot_loop_edges = sum(1 for edge in snapshot.edges if edge.edge_type == "loop")
        print(
            "[Pose Graph Final Started] "
            f"frames={len(snapshot.frame_ids)} edges={len(snapshot.edges)} "
            f"loops={snapshot_loop_edges} "
            f"max_iter={self.config.pose_opt_max_nfev}"
        )
        final_start = time.perf_counter()
        try:
            result = optimize_pose_graph(
                snapshot,
                max_nfev=int(self.config.pose_opt_max_nfev),
            )
        except Exception as exc:
            event = {
                "event": "pose_graph_failed",
                "phase": "final",
                "error": str(exc),
                "frame_count": int(self.state.frame_count()),
                "num_snapshot_frames": int(len(snapshot.frame_ids)),
                "num_edges": int(len(snapshot.edges)),
                "snapshot_build_time_sec": snapshot_build_time,
                "final_total_time_sec": time.perf_counter() - final_start,
            }
            print(
                "[Pose Graph Failed] "
                f"phase=final frames={event['num_snapshot_frames']} "
                f"edges={event['num_edges']} error={event['error']}"
            )
        else:
            event = self._merge_pose_graph_result(result)
            event["phase"] = "final"
            event["loop_edges"] = int(snapshot_loop_edges)
            event["snapshot_build_time_sec"] = snapshot_build_time
            event["final_total_time_sec"] = time.perf_counter() - final_start
            print(
                "[Pose Graph Final Finished] "
                f"frames={event.get('num_snapshot_frames')} edges={result.summary.get('num_edges')} "
                f"loops={event.get('loop_edges')} "
                f"pgo={float(event.get('pgo_time_sec') or 0.0):.2f}s "
                f"pose_update={float(event.get('pose_update_time_sec') or 0.0):.2f}s "
                f"point_refusion=disabled "
                f"total={float(event.get('final_total_time_sec') or 0.0):.2f}s"
            )
        self.pose_graph_events.append(event)
        events.append(event)
        self.runtime_stats["pose_graph_finalize_time_sec"] = time.perf_counter() - finalize_start
        return events

    @staticmethod
    def _print_pose_graph_discard(event: Dict[str, object]) -> None:
        summary = event.get("summary") if isinstance(event.get("summary"), dict) else {}
        print(
            "[Pose Graph Discarded] "
            f"reason={event.get('reason')} "
            f"snapshot_frames={event.get('snapshot_frame_count')} "
            f"current_frames={event.get('current_frame_count')} "
            f"tail={event.get('tail_frame_count', 0)} "
            f"cost={float(summary.get('initial_cost') or 0.0):.6f}->"
            f"{float(summary.get('final_cost') or 0.0):.6f}"
        )

    @staticmethod
    def _public_pose_graph_summary(summary: Dict[str, object]) -> Dict[str, object]:
        public_summary = dict(summary)
        public_summary.pop("backend", None)
        public_summary.pop("model", None)
        return public_summary

    @staticmethod
    def _is_near_identity_delta(delta: np.ndarray) -> bool:
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape != (4, 4) or not np.isfinite(delta).all():
            return False
        translation_norm = float(np.linalg.norm(delta[:3, 3]))
        rotation_deg = PoseGraphMixin._delta_rotation_deg(delta)
        return translation_norm <= 1e-8 and rotation_deg <= 1e-6

    @staticmethod
    def _delta_rotation_deg(delta: np.ndarray) -> float:
        delta = np.asarray(delta, dtype=np.float64)
        rotation = delta[:3, :3]
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        trace = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(trace)))

    @staticmethod
    def _value_stats(values: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
        if arr.size == 0:
            return {"mean": 0.0, "median": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
        }

    def _apply_optimized_pose_update(
        self,
        *,
        frame_id: str,
        new_pose: np.ndarray,
    ) -> None:
        self.state.set_optimized_pose(frame_id, new_pose)

    def _make_pose_graph_debug_payload(self, result: PoseGraphResult) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "frame_ids": list(result.frame_ids),
            "initial_centers": {},
            "refined_centers": {},
            "summary": self._public_pose_graph_summary(result.summary),
        }
        initial_centers = payload["initial_centers"]
        refined_centers = payload["refined_centers"]
        assert isinstance(initial_centers, dict)
        assert isinstance(refined_centers, dict)
        for idx, frame_id in enumerate(result.frame_ids):
            initial_centers[frame_id] = np.asarray(result.cam2world_initial[idx], dtype=np.float64)[:3, 3].tolist()
            refined_centers[frame_id] = np.asarray(result.cam2world_refined[idx], dtype=np.float64)[:3, 3].tolist()
        return payload

from concurrent.futures import Future, ThreadPoolExecutor
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from on_the_fly3r.pose_graph import PoseGraphEdge, PoseGraphMixin
from on_the_fly3r.output import PoseGraphOutputMixin

from .config import ReconstructionConfig
from .vfm_adapter import VFMInferenceRunner
from on_the_fly3r.alignment import AlignmentMixin
from on_the_fly3r.retry import RetryMixin
from on_the_fly3r.validation import ValidationMixin
from on_the_fly3r.fusion import FusionMixin
from on_the_fly3r.pointcloud import PointCloudMixin
from on_the_fly3r.preprocessing import PreprocessingMixin
from on_the_fly3r.retrieval import (
    RetrievalPlanningMixin,
    BatchPlan,
    RetrievalManager,
)

from on_the_fly3r.state import ReconstructionState
from .types import (
    AlignmentResult,
    FrameReconstruction,
    SubsetInferenceResult,
)
class IncrementalReconstructor(RetrievalPlanningMixin, PreprocessingMixin, AlignmentMixin, ValidationMixin, RetryMixin, FusionMixin, PointCloudMixin, PoseGraphMixin, PoseGraphOutputMixin):
    """
    Framework for progressive multi-model reconstruction.

    Pipeline:
    1. Bootstrap the model from the first N images.
    2. For each incoming batch, retrieve top-k frames from the already reconstructed set.
    3. Run the selected VFM on the retrieved frames plus the new batch.
    4. Estimate a transform from the local subset coordinate system to the global one.
    5. Transform only the new frames into the global model and fuse their point clouds.
    """

    def __init__(
        self,
        config: ReconstructionConfig,
        model_args,
        retrieval_feature_fn: Optional[Callable[[str], np.ndarray]] = None,
        alignment_fn: Optional[
            Callable[[SubsetInferenceResult, List[FrameReconstruction], str], AlignmentResult]
        ] = None,
    ) -> None:
        self.config = config
        self.runner = VFMInferenceRunner(config, model_args)
        self.state = ReconstructionState()
        self.retrieval_manager = RetrievalManager(config, retrieval_feature_fn)
        self.index = self.retrieval_manager.index
        self.retrieval_encoder = self.retrieval_manager.encoder
        self.retrieval_feature_fn = self.retrieval_manager.feature_fn
        self.retrieval_feature_cache = self.retrieval_manager.feature_cache
        image_prefetch_available = self.config.image_preprocess_prefetch_count > 0
        image_prefetch_reason = "ok" if image_prefetch_available else "zero_count"
        self._image_preprocess_prefetch_executor = (
            ThreadPoolExecutor(
                max_workers=int(self.config.image_preprocess_prefetch_workers),
                thread_name_prefix="image-preprocess-prefetch",
            )
            if image_prefetch_available
            else None
        )
        self._image_preprocess_prefetch_futures: List[Future[Dict[str, int]]] = []
        self._image_preprocess_prefetch_stats: Dict[str, object] = {
            "enabled": bool(image_prefetch_available),
            "reason": image_prefetch_reason,
            "submitted": 0,
            "completed": 0,
            "skipped_cached": 0,
            "skipped_pending": 0,
            "errors": 0,
            "current_pending": 0,
        }
        self.pose_graph_edges: List[PoseGraphEdge] = []
        self.pose_graph_events: List[Dict[str, object]] = []
        self._last_pose_graph_debug_payload: Optional[Dict[str, object]] = None
        self._continuity_translation_steps: List[float] = []
        self._continuity_rotation_steps_deg: List[float] = []
        self.alignment_fn = alignment_fn or self._estimate_alignment
        self.bootstrap_log: Dict[str, object] = {}
        self.dataset_spacing_reference: Optional[float] = None
        self.validation_translation_threshold: Optional[float] = None
        self.runtime_stats: Dict[str, object] = {
            "total_time_sec": 0.0,
            "bootstrap_time_sec": 0.0,
            "processing_time_sec": 0.0,
        }

    def shutdown(self) -> None:
        self.runner.clear_image_preprocess_cache()
        self.retrieval_manager.shutdown()
        self._poll_image_preprocess_prefetch()
        if self._image_preprocess_prefetch_executor is not None:
            self._image_preprocess_prefetch_executor.shutdown(wait=False, cancel_futures=False)
            self._image_preprocess_prefetch_executor = None


    def bootstrap(self, image_paths: Sequence[str]) -> None:
        if len(image_paths) < self.config.init_window:
            raise ValueError(
                f"Need at least {self.config.init_window} images for initialization, "
                f"but only got {len(image_paths)}."
            )

        init_paths = list(image_paths[: self.config.init_window])
        bootstrap_start = time.perf_counter()
        subset, inference_stats = self._measure_stage(
            lambda: self.runner.infer_subset(init_paths),
            device_type=self.config.inference_device,
        )
        if subset.profile:
            inference_stats["profile"] = dict(subset.profile)
        bootstrap_spacing_reference = self._estimate_dataset_spacing_reference_from_dense_batch(
            subset.world_points,
            subset.world_points_conf,
        )

        retrieval_time_sec = 0.0
        fusion_time_sec = 0.0
        for idx, frame_id in enumerate(subset.frame_ids):
            feature, retrieval_stats = self._measure_stage(
                lambda path=subset.image_paths[idx]: self._get_cached_retrieval_feature(path),
                device_type=self.config.retrieval_device,
            )
            retrieval_time_sec += retrieval_stats["time_sec"]

            fusion_start = time.perf_counter()
            frame = self._build_global_frame(
                frame_id=frame_id,
                image_path=subset.image_paths[idx],
                cam2world=subset.cam2world[idx],
                intrinsic=subset.intrinsic[idx],
                world_points=subset.world_points[idx],
                world_points_conf=subset.world_points_conf[idx],
                image=subset.images[idx],
                retrieval_vector=feature,
                transform=np.eye(4, dtype=np.float32),
            )
            fusion_time_sec += time.perf_counter() - fusion_start
            self.state.add_frame(frame)
            self.index.add(frame_id, feature)

        self.dataset_spacing_reference = bootstrap_spacing_reference
        if self.dataset_spacing_reference is None:
            self.dataset_spacing_reference = self._estimate_dataset_spacing_reference(
                frames=[self.state.get_frame(frame_id) for frame_id in subset.frame_ids]
            )
        if self.dataset_spacing_reference is not None:
            self.validation_translation_threshold = (
                float(self.config.validation_translation_spacing_multiplier)
                * float(self.dataset_spacing_reference)
            )
        if self.config.enable_pose_graph_optimization:
            edge_count_before = len(self.pose_graph_edges)
            self._record_local_continuity_edges(
                fresh_frame_ids=subset.frame_ids[1:],
                scored_neighbors_per_frame={},
            )
            added_edges = len(self.pose_graph_edges) - edge_count_before
            if added_edges > 0:
                self.pose_graph_events.append(
                    {
                        "event": "bootstrap_edges_recorded",
                        "added_edges": int(added_edges),
                        "total_edges": int(len(self.pose_graph_edges)),
                        "frame_count": int(self.state.frame_count()),
                    }
                )
        bootstrap_total = time.perf_counter() - bootstrap_start
        self.runtime_stats["bootstrap_time_sec"] = bootstrap_total
        self.bootstrap_log = {
            "num_init_frames": len(subset.frame_ids),
            "inference": inference_stats,
            "retrieval": {
                "time_sec": retrieval_time_sec,
            },
            "fusion_time_sec": fusion_time_sec,
            "dataset_spacing_reference": self.dataset_spacing_reference,
            "validation_translation_threshold": self.validation_translation_threshold,
            "total_time_sec": bootstrap_total,
        }







































    def process_next_batch(
        self,
        image_paths: Sequence[str],
        *,
        batch_idx: Optional[int] = None,
        batch_plan: Optional[BatchPlan] = None,
        prefetch_next_paths: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        batch_start = time.perf_counter()
        pose_graph_events: List[Dict[str, object]] = []
        requested_paths = list(batch_plan.image_paths if batch_plan is not None else image_paths)
        requested_frame_ids = [Path(path).stem for path in requested_paths]

        fresh_items = [(frame_id, path) for frame_id, path in zip(requested_frame_ids, requested_paths) if not self.state.has_frame(frame_id)]
        skipped_existing = [frame_id for frame_id, path in zip(requested_frame_ids, requested_paths) if self.state.has_frame(frame_id)]
        if not fresh_items:
            return {
                "frame_ids": requested_frame_ids,
                "status": "skipped_existing",
                "skipped_existing": skipped_existing,
                "pose_graph_events": pose_graph_events,
                "total_time_sec": time.perf_counter() - batch_start,
            }

        fresh_frame_ids = [frame_id for frame_id, _ in fresh_items]
        fresh_paths = [path for _, path in fresh_items]

        if batch_plan is None:
            def retrieval_step():
                query_feature_map: Dict[str, np.ndarray] = {}
                scored_neighbors_per_frame: Dict[str, List[tuple[str, float]]] = {}
                for frame_id, image_path in zip(fresh_frame_ids, fresh_paths):
                    context = self._prepare_query_context(image_path)
                    query_feature_map[frame_id] = context.feature
                    scored_neighbors_per_frame[frame_id] = context.scored_neighbors
                ranked_neighbors, selected_neighbors = self._aggregate_scored_neighbors(scored_neighbors_per_frame)
                return query_feature_map, scored_neighbors_per_frame, ranked_neighbors, selected_neighbors

            (
                query_feature_map,
                scored_neighbors_per_frame,
                ranked_neighbors,
                neighbor_ids,
            ), retrieval_stats = self._measure_stage(
                retrieval_step,
                device_type=self.config.retrieval_device,
            )
        else:
            query_feature_map = {
                frame_id: np.asarray(batch_plan.query_features[frame_id], dtype=np.float32)
                for frame_id in fresh_frame_ids
            }
            scored_neighbors_per_frame = {
                frame_id: list(batch_plan.scored_neighbors_per_frame.get(frame_id, []))
                for frame_id in fresh_frame_ids
            }
            ranked_neighbors = list(batch_plan.ranked_neighbors)
            neighbor_ids = list(batch_plan.selected_neighbors)
            retrieval_stats = dict(batch_plan.retrieval_stats)

        query_features = [query_feature_map[frame_id] for frame_id in fresh_frame_ids]
        if len(neighbor_ids) < self.config.min_topk_for_alignment:
            log = {
                "frame_ids": fresh_frame_ids,
                "status": "skipped_not_enough_neighbors",
                "neighbor_count": len(neighbor_ids),
                "retrieval": retrieval_stats,
                "retrieved_neighbors": ranked_neighbors,
                "skipped_existing": skipped_existing,
                "pose_graph_events": pose_graph_events,
                "formation": {
                    "reason": batch_plan.formation_reason,
                    "diagnostics": batch_plan.formation_diagnostics,
                    "blocked_frame_id": batch_plan.blocked_frame_id,
                } if batch_plan is not None else None,
                "total_time_sec": time.perf_counter() - batch_start,
            }
            return log

        neighbor_frames = self.state.get_frames(neighbor_ids)
        subset_paths = [frame.image_path for frame in neighbor_frames] + fresh_paths
        subset, inference_stats = self._measure_stage(
            lambda: self.runner.infer_subset(subset_paths),
            device_type=self.config.inference_device,
        )
        if subset.profile:
            inference_stats["profile"] = dict(subset.profile)
        if prefetch_next_paths:
            self.prefetch_retrieval_features(prefetch_next_paths)
            self.prefetch_image_preprocess(prefetch_next_paths)

        alignment_attempt = self.run_alignment_attempt(
            subset=subset,
            neighbor_frames=neighbor_frames,
            new_frame_label=",".join(fresh_frame_ids),
            validate=self.config.enable_alignment_validation,
        )
        if alignment_attempt["alignment_error"] is not None:
            exc = alignment_attempt["alignment_error"]
            alignment_time_sec = alignment_attempt["alignment_time_sec"]
            batch_label = batch_idx if batch_idx is not None else "?"
            print(
                f"[Alignment Failed] batch={batch_label} "
                "mode=point_sim3 "
                f"frames={','.join(fresh_frame_ids)} "
                f"reason={exc}"
            )
            log = {
                "frame_ids": fresh_frame_ids,
                "status": "skipped_alignment_failed",
                "batch_size": len(fresh_frame_ids),
                "skipped_existing": skipped_existing,
                "neighbors": neighbor_ids,
                "retrieved_neighbors": ranked_neighbors[: self.config.retrieval_topk],
                "retrieval": retrieval_stats,
                "inference": inference_stats,
                "alignment_time_sec": alignment_time_sec,
                "diagnostics": {
                    "mode": "point_sim3",
                    "error": str(exc),
                    "new_frame_id": ",".join(fresh_frame_ids),
                },
                "pose_graph_events": pose_graph_events,
                "formation": {
                    "reason": batch_plan.formation_reason,
                    "diagnostics": batch_plan.formation_diagnostics,
                    "blocked_frame_id": batch_plan.blocked_frame_id,
                } if batch_plan is not None else None,
                "total_time_sec": time.perf_counter() - batch_start,
            }
            return log
        alignment = alignment_attempt["alignment"]
        alignment_time_sec = alignment_attempt["alignment_time_sec"]

        validation = alignment_attempt["validation"]
        validation_time_sec = alignment_attempt["validation_time_sec"]
        if validation is not None:
            if not validation["accepted"]:
                batch_label = batch_idx if batch_idx is not None else "?"
                metrics = validation.get("metrics", {})
                print(
                    f"[Validation Failed] batch={batch_label} "
                    "mode=point_sim3 "
                    f"frames={','.join(fresh_frame_ids)} "
                    f"reason={validation['reason']} "
                    f"median_t={metrics.get('median_translation_residual', float('nan')):.6f} "
                    f"max_t={metrics.get('max_translation_residual', float('nan')):.6f} "
                    f"median_r={metrics.get('median_rotation_residual_deg', float('nan')):.3f}deg "
                    f"max_r={metrics.get('max_rotation_residual_deg', float('nan')):.3f}deg"
                )
                retry_result = None
                register_retry_log = None
                if self.config.enable_register_ref_pruning_retry:
                    retry_result = self._attempt_two_stage_ref_pruning_retry(
                        subset=subset,
                        neighbor_frames=neighbor_frames,
                        fresh_paths=fresh_paths,
                        fresh_frame_ids=fresh_frame_ids,
                        initial_validation=validation,
                    )
                    register_retry_log = retry_result.get("retry_log") if retry_result is not None else None
                if retry_result is not None and retry_result.get("validation", {}).get("accepted", False):
                    retry_subset = retry_result["subset"]
                    retry_neighbor_frames = retry_result["neighbor_frames"]
                    retry_alignment = retry_result["alignment"]
                    accepted_inference_stats = dict(retry_result["inference"])
                    if accepted_inference_stats.get("reused_initial_subset", False):
                        accepted_inference_stats = {
                            **inference_stats,
                            "reused_initial_subset": True,
                        }
                    fusion_time_sec, added_frames, fusion_profile = self._append_aligned_batch_to_state(
                        subset=retry_subset,
                        neighbor_frames=retry_neighbor_frames,
                        fresh_frame_ids=fresh_frame_ids,
                        fresh_paths=fresh_paths,
                        query_features=query_features,
                        scored_neighbors_per_frame=scored_neighbors_per_frame,
                        alignment=retry_alignment,
                    )
                    subset.vfm_register_tokens = None
                    accepted_stage = register_retry_log.get("accepted_stage", "unknown") if register_retry_log else "unknown"
                    print(
                        f"[Ref Pruning Retry Added] batch={batch_label} "
                        f"frames={','.join(fresh_frame_ids)} "
                        f"stage={accepted_stage} "
                        f"dropped={','.join(register_retry_log.get('dropped_neighbors', []))}"
                    )
                    return {
                        "frame_ids": fresh_frame_ids,
                        "status": "added",
                        "batch_size": len(fresh_frame_ids),
                        "skipped_existing": skipped_existing,
                        "neighbors": retry_result["neighbor_ids"],
                        "retrieved_neighbors": ranked_neighbors[: self.config.retrieval_topk],
                        "inliers": int(len(retry_alignment.inlier_indices)),
                        "retrieval": retrieval_stats,
                        "inference": accepted_inference_stats,
                        "initial_inference": inference_stats,
                        "alignment_time_sec": retry_result["alignment_time_sec"],
                        "initial_alignment_time_sec": alignment_time_sec,
                        "initial_validation_time_sec": validation_time_sec,
                        "fusion_time_sec": fusion_time_sec,
                        "fusion_profile": fusion_profile if self.config.enable_detailed_profiling else None,
                        "diagnostics": retry_alignment.diagnostics,
                        "initial_diagnostics": alignment.diagnostics,
                        "validation": retry_result["validation"],
                        "initial_validation": validation,
                        "register_ref_pruning_retry": register_retry_log,
                        "pose_graph_events": pose_graph_events,
                        "formation": {
                            "reason": batch_plan.formation_reason,
                            "diagnostics": batch_plan.formation_diagnostics,
                            "blocked_frame_id": batch_plan.blocked_frame_id,
                        } if batch_plan is not None else None,
                        "added_frames": added_frames,
                        "total_time_sec": time.perf_counter() - batch_start,
                    }

                subset.vfm_register_tokens = None
                log = {
                    "frame_ids": fresh_frame_ids,
                    "status": "skipped_failed_validation",
                    "batch_size": len(fresh_frame_ids),
                    "skipped_existing": skipped_existing,
                    "neighbors": neighbor_ids,
                    "retrieved_neighbors": ranked_neighbors[: self.config.retrieval_topk],
                    "retrieval": retrieval_stats,
                    "inference": inference_stats,
                    "alignment_time_sec": alignment_time_sec,
                    "validation_time_sec": validation_time_sec,
                    "diagnostics": alignment.diagnostics,
                    "validation": validation,
                    "register_ref_pruning_retry": register_retry_log,
                    "pose_graph_events": pose_graph_events,
                    "formation": {
                        "reason": batch_plan.formation_reason,
                        "diagnostics": batch_plan.formation_diagnostics,
                        "blocked_frame_id": batch_plan.blocked_frame_id,
                    } if batch_plan is not None else None,
                    "total_time_sec": time.perf_counter() - batch_start,
                }
                return log

        fusion_time_sec, added_frames, fusion_profile = self._append_aligned_batch_to_state(
            subset=subset,
            neighbor_frames=neighbor_frames,
            fresh_frame_ids=fresh_frame_ids,
            fresh_paths=fresh_paths,
            query_features=query_features,
            scored_neighbors_per_frame=scored_neighbors_per_frame,
            alignment=alignment,
        )
        log = {
            "frame_ids": fresh_frame_ids,
            "status": "added",
            "batch_size": len(fresh_frame_ids),
            "skipped_existing": skipped_existing,
            "neighbors": neighbor_ids,
            "retrieved_neighbors": ranked_neighbors[: self.config.retrieval_topk],
            "inliers": int(len(alignment.inlier_indices)),
            "retrieval": retrieval_stats,
            "inference": inference_stats,
            "alignment_time_sec": alignment_time_sec,
            "validation_time_sec": validation_time_sec,
            "fusion_time_sec": fusion_time_sec,
            "fusion_profile": fusion_profile if self.config.enable_detailed_profiling else None,
            "diagnostics": alignment.diagnostics,
            "validation": validation,
            "pose_graph_events": pose_graph_events,
            "formation": {
                "reason": batch_plan.formation_reason,
                "diagnostics": batch_plan.formation_diagnostics,
                "blocked_frame_id": batch_plan.blocked_frame_id,
            } if batch_plan is not None else None,
            "added_frames": added_frames,
            "total_time_sec": time.perf_counter() - batch_start,
        }
        return log

    def run_dataset(
        self,
        image_paths: Sequence[str],
        batch_callback: Optional[Callable[[int, int, Dict[str, object]], None]] = None,
    ) -> List[Dict[str, object]]:
        ordered_paths = list(image_paths)
        total_start = time.perf_counter()
        self.bootstrap(ordered_paths[: self.config.init_window])
        processing_start = time.perf_counter()

        logs: List[Dict[str, object]] = []
        remaining = ordered_paths[self.config.init_window :]
        total_remaining_images = len(remaining)

        batch_idx = 1
        cursor = 0
        while cursor < len(remaining):
            batch_plan, cursor = self._form_next_dynamic_batch(
                remaining_paths=remaining,
                start_idx=cursor,
            )
            message = (
                f"[Dynamic Batch] idx={batch_idx} size={len(batch_plan.frame_ids)} "
                f"frames={','.join(batch_plan.frame_ids)} reason={batch_plan.formation_reason}"
            )
            if batch_plan.blocked_frame_id:
                message += f" blocked={batch_plan.blocked_frame_id}"
            print(message)
            prefetch_next_paths = remaining[cursor: cursor + self.config.retrieval_prefetch_count]
            log = self.process_next_batch(
                batch_plan.image_paths,
                batch_idx=batch_idx,
                batch_plan=batch_plan,
                prefetch_next_paths=prefetch_next_paths,
            )
            logs.append(log)
            if batch_callback is not None:
                batch_callback(batch_idx, total_remaining_images, log)
            batch_idx += 1
        self.runtime_stats["processing_time_sec"] = time.perf_counter() - processing_start
        self.runtime_stats["total_time_sec"] = time.perf_counter() - total_start
        return logs

    def get_runtime_summary(self) -> Dict[str, object]:
        self._poll_image_preprocess_prefetch()
        return {
            **self.runtime_stats,
            "bootstrap": self.bootstrap_log,
            "pose_graph": {
                "enabled": bool(self.config.enable_pose_graph_optimization),
                "mode": "final",
                "num_edges": int(len(self.pose_graph_edges)),
                "events": list(self.pose_graph_events),
            },
            "image_preprocess_cache": self.runner.image_preprocess_cache_stats(),
            "image_preprocess_prefetch": dict(self._image_preprocess_prefetch_stats),
        }





    def export_camera_poses(
        self,
        image_paths: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        return self.state.export_camera_poses(image_paths=image_paths)

















    def _measure_stage(self, fn: Callable[[], object], device_type: str) -> tuple[object, Dict[str, float]]:
        use_cuda = (
            self.config.enable_profiling_sync
            and str(device_type).lower().startswith("cuda")
            and torch.cuda.is_available()
        )
        if use_cuda:
            torch.cuda.synchronize(device_type)
        start = time.perf_counter()
        result = fn()
        if use_cuda:
            torch.cuda.synchronize(device_type)
        elapsed = time.perf_counter() - start
        return result, {"time_sec": elapsed}

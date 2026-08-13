"""Fusion of aligned VFM batches into reconstruction state."""

from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np

from .types import AlignmentResult, FrameReconstruction, SubsetInferenceResult


class FusionMixin:
    def _append_aligned_batch_to_state(
        self,
        *,
        subset: SubsetInferenceResult,
        neighbor_frames: List[FrameReconstruction],
        fresh_frame_ids: Sequence[str],
        fresh_paths: Sequence[str],
        query_features: Sequence[np.ndarray],
        scored_neighbors_per_frame: Dict[str, List[tuple[str, float]]],
        alignment: AlignmentResult,
    ) -> tuple[float, List[Dict[str, object]], Dict[str, object]]:
        subset.vfm_register_tokens = None
        offset = len(neighbor_frames)
        fusion_time_sec = 0.0
        added_frames: List[Dict[str, object]] = []
        fusion_profile: Dict[str, object] = {"frames": []} if self.config.enable_detailed_profiling else {}
        for local_offset, (frame_id, image_path, query_feature) in enumerate(zip(fresh_frame_ids, fresh_paths, query_features)):
            local_index = offset + local_offset
            fusion_start = time.perf_counter()
            frame_profile: Dict[str, object] = {"frame_id": frame_id} if self.config.enable_detailed_profiling else {}
            build_frame_start = time.perf_counter()
            global_frame = self._build_global_frame(
                frame_id=frame_id,
                image_path=image_path,
                cam2world=subset.cam2world[local_index],
                intrinsic=subset.intrinsic[local_index],
                world_points=subset.world_points[local_index],
                world_points_conf=subset.world_points_conf[local_index],
                image=subset.images[local_index],
                retrieval_vector=query_feature,
                transform=alignment.transform_local_to_global,
            )
            if self.config.enable_detailed_profiling:
                frame_profile["build_global_frame_sec"] = time.perf_counter() - build_frame_start
                if global_frame.metadata.get("fusion_profile"):
                    frame_profile["build_profile"] = global_frame.metadata.get("fusion_profile")
            global_frame.metadata["scored_neighbors"] = list(scored_neighbors_per_frame.get(frame_id, []))
            frame_total_sec = time.perf_counter() - fusion_start
            fusion_time_sec += frame_total_sec
            self.state.add_frame(global_frame)
            self.index.add(frame_id, query_feature)
            added_frame = {
                "frame_id": frame_id,
                "fused_points_count": int(global_frame.metadata.get("fused_points_count", 0)),
                "fusion_raw_points_count": int(
                    global_frame.metadata.get(
                        "fusion_raw_points_count",
                        global_frame.metadata.get("fused_points_count", 0),
                    )
                ),
                "per_frame_neighbors": scored_neighbors_per_frame.get(frame_id, []),
            }
            if self.config.enable_detailed_profiling:
                frame_profile["total_sec"] = frame_total_sec
                added_frame["fusion_profile"] = frame_profile
                fusion_profile["frames"].append(frame_profile)
            added_frames.append(added_frame)

        pose_graph_record_start = time.perf_counter()
        self._record_pose_graph_observations(
            subset=subset,
            neighbor_frames=neighbor_frames,
            fresh_frame_ids=fresh_frame_ids,
            alignment=alignment,
            scored_neighbors_per_frame=scored_neighbors_per_frame,
        )
        if self.config.enable_detailed_profiling:
            fusion_profile["pose_graph_record_sec"] = time.perf_counter() - pose_graph_record_start
        return fusion_time_sec, added_frames, fusion_profile

    def _build_global_frame(
        self,
        frame_id: str,
        image_path: str,
        cam2world: np.ndarray,
        intrinsic: np.ndarray,
        world_points: np.ndarray,
        world_points_conf: np.ndarray,
        image: np.ndarray,
        retrieval_vector: Optional[np.ndarray],
        transform: np.ndarray,
    ) -> FrameReconstruction:
        profile: Dict[str, float] = {}
        stage_start = time.perf_counter()
        cam2world_global = self._transform_pose(transform, cam2world)
        if self.config.enable_detailed_profiling:
            profile["transform_pose_sec"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        world_points_global = self._transform_points(transform, world_points)
        if self.config.enable_detailed_profiling:
            profile["transform_dense_points_sec"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        fused_points, fused_colors, fused_weights = self._prepare_frame_points_for_fusion(
            world_points=world_points_global,
            world_points_conf=world_points_conf,
            image=image,
            conf_threshold=self.config.fusion_conf_threshold,
        )
        if self.config.enable_detailed_profiling:
            profile["prepare_points_for_fusion_sec"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        fused_chunk_index = self.state.append_fused_points(fused_points, fused_colors, fused_weights)
        if self.config.enable_detailed_profiling:
            profile["append_fused_points_sec"] = time.perf_counter() - stage_start

        metadata = {
            "conf_threshold": self.config.fusion_conf_threshold,
            "fused_points_count": int(fused_points.shape[0]),
            "global_points_chunk_index": fused_chunk_index,
            "image_shape": tuple(int(v) for v in image.shape[:2]),
            "viewer_thumbnail": self._make_image_thumbnail(image),
        }
        metadata["alignment_compact_reference"] = self._build_compact_alignment_reference(
            world_points=world_points_global,
            world_points_conf=world_points_conf,
            conf_threshold=self.config.alignment_point_conf_threshold,
            max_points=self.config.alignment_compact_max_points,
            grid_cell_size=self.config.alignment_compact_grid_cell_size,
            topk_per_cell=self.config.alignment_grid_topk_per_cell,
        )
        if self.config.enable_detailed_profiling:
            metadata["fusion_profile"] = profile
        return FrameReconstruction(
            frame_id=frame_id,
            image_path=image_path,
            cam2world=cam2world_global.astype(np.float32, copy=False),
            intrinsic=intrinsic.astype(np.float32, copy=False),
            world_points=None,
            world_points_conf=None,
            image=None,
            retrieval_vector=retrieval_vector,
            metadata=metadata,
        )

    @staticmethod
    def _prepare_frame_points_for_fusion(
        world_points: np.ndarray,
        world_points_conf: np.ndarray,
        image: np.ndarray,
        conf_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid = np.isfinite(world_points).all(axis=-1)
        valid &= np.isfinite(world_points_conf)
        valid &= world_points_conf >= conf_threshold

        points = world_points[valid].reshape(-1, 3).astype(np.float32, copy=False)
        colors = image[valid].reshape(-1, 3).astype(np.uint8, copy=False)
        weights = world_points_conf[valid].reshape(-1).astype(np.float32, copy=False)
        return points, colors, weights

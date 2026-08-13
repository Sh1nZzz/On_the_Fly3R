"""Alignment and compact-reference operations used by the reconstruction pipeline."""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from .types import AlignmentResult, FrameReconstruction, SubsetInferenceResult


class AlignmentMixin:
    def run_alignment_attempt(
        self,
        *,
        subset: SubsetInferenceResult,
        neighbor_frames: List[FrameReconstruction],
        new_frame_label: str,
        validate: bool,
    ) -> Dict[str, object]:
        """Run one alignment and optional validation with consistent timing."""
        alignment_start = time.perf_counter()
        try:
            alignment = self.alignment_fn(subset, neighbor_frames, new_frame_label)
        except Exception as exc:
            return {
                "alignment": None,
                "alignment_error": exc,
                "alignment_time_sec": time.perf_counter() - alignment_start,
                "validation": None,
                "validation_time_sec": 0.0,
            }
        alignment_time_sec = time.perf_counter() - alignment_start

        validation = None
        validation_time_sec = 0.0
        if validate:
            validation_start = time.perf_counter()
            validation = self._validate_alignment_result(
                subset=subset,
                neighbor_frames=neighbor_frames,
                alignment=alignment,
            )
            validation_time_sec = time.perf_counter() - validation_start
        return {
            "alignment": alignment,
            "alignment_error": None,
            "alignment_time_sec": alignment_time_sec,
            "validation": validation,
            "validation_time_sec": validation_time_sec,
        }

    def _estimate_alignment(
        self,
        subset: SubsetInferenceResult,
        neighbor_frames: List[FrameReconstruction],
        new_frame_id: str,
    ) -> AlignmentResult:
        if len(neighbor_frames) <= 0:
            raise ValueError("No neighbor frames available for alignment.")
        if len(neighbor_frames) >= len(subset.frame_ids):
            raise ValueError("Subset must contain neighbor frames followed by at least one new frame.")

        transform, inliers, diagnostics = self._estimate_point_sim3_transform(
            subset=subset,
            neighbor_frames=neighbor_frames,
            use_scale=self.config.alignment_use_scale,
            conf_threshold=self.config.alignment_point_conf_threshold,
            grid_conf_min=self.config.alignment_grid_conf_min,
            fine_residual_mad_scale=self.config.alignment_fine_residual_mad_scale,
        )

        return AlignmentResult(
            transform_local_to_global=transform,
            inlier_indices=inliers,
            diagnostics={**diagnostics, "new_frame_id": new_frame_id},
        )

    @staticmethod
    def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
        shape = points.shape
        points_flat = points.reshape(-1, 3)
        valid = np.isfinite(points_flat).all(axis=1)
        out = np.full_like(points_flat, np.nan, dtype=np.float32)
        if np.any(valid):
            points_valid = points_flat[valid]
            points_h = np.concatenate(
                [points_valid, np.ones((points_valid.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            transformed = (transform @ points_h.T).T
            out[valid] = transformed[:, :3] / np.clip(transformed[:, 3:4], 1e-8, None)
        return out.reshape(shape)

    @staticmethod
    def _transform_pose(transform: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
        transform = np.asarray(transform, dtype=np.float64)
        cam2world = np.asarray(cam2world, dtype=np.float64)

        sim3_linear = transform[:3, :3]
        sim3_translation = transform[:3, 3]
        scale = float(np.mean(np.linalg.norm(sim3_linear, axis=0)))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0

        rotation = sim3_linear / scale
        U, _, Vt = np.linalg.svd(rotation)
        rotation = U @ Vt
        if np.linalg.det(rotation) < 0:
            U[:, -1] *= -1.0
            rotation = U @ Vt

        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = rotation @ cam2world[:3, :3]
        out[:3, 3] = sim3_linear @ cam2world[:3, 3] + sim3_translation
        return out.astype(np.float32)

    @staticmethod
    def _make_image_thumbnail(image: np.ndarray, max_width: int = 88, max_height: int = 56) -> np.ndarray:
        image_np = np.asarray(image, dtype=np.uint8)
        if image_np.ndim != 3 or image_np.shape[-1] != 3 or image_np.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)
        height, width = int(image_np.shape[0]), int(image_np.shape[1])
        if height <= 0 or width <= 0:
            return np.empty((0, 0, 3), dtype=np.uint8)
        scale = min(float(max_width) / float(width), float(max_height) / float(height), 1.0)
        out_w = max(1, int(round(width * scale)))
        out_h = max(1, int(round(height * scale)))
        if out_w == width and out_h == height:
            return image_np.copy()
        rows = np.linspace(0, height - 1, num=out_h, dtype=np.int64)
        cols = np.linspace(0, width - 1, num=out_w, dtype=np.int64)
        return image_np[rows][:, cols].copy()

    @staticmethod
    def _robust_residual_threshold(
        residuals: np.ndarray,
        mad_scale: float,
        minimum_slack: float = 1e-6,
    ) -> float:
        residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
        if residuals.size == 0:
            return float("inf")
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        sigma = 1.4826 * mad
        return median + mad_scale * max(sigma, minimum_slack)

    @staticmethod
    def _select_grid_conf_topk_indices(
        weights: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        *,
        max_corr: int,
        cell_size: int,
        topk_per_cell: int,
    ) -> np.ndarray:
        if weights.size == 0:
            return np.empty((0,), dtype=np.int32)

        cell_size = max(1, int(cell_size))
        topk_per_cell = max(1, int(topk_per_cell))
        cell_rows = rows.astype(np.int64, copy=False) // cell_size
        cell_cols = cols.astype(np.int64, copy=False) // cell_size
        order = np.lexsort(
            (
                np.arange(weights.shape[0], dtype=np.int64),
                -np.asarray(weights, dtype=np.float64),
                cell_cols,
                cell_rows,
            )
        )
        sorted_cell_rows = cell_rows[order]
        sorted_cell_cols = cell_cols[order]
        first = np.ones((order.shape[0],), dtype=bool)
        first[1:] = (
            (sorted_cell_rows[1:] != sorted_cell_rows[:-1])
            | (sorted_cell_cols[1:] != sorted_cell_cols[:-1])
        )
        group_ids = np.cumsum(first.astype(np.int64)) - 1
        group_starts = np.flatnonzero(first)
        ranks = np.arange(order.shape[0], dtype=np.int64) - group_starts[group_ids]
        selected = order[ranks < topk_per_cell]

        if max_corr is not None and selected.shape[0] > max_corr:
            top = np.argsort(-weights[selected])[:max_corr]
            selected = selected[top]
        return selected.astype(np.int32, copy=False)

    @classmethod
    def _build_compact_alignment_reference(
        cls,
        *,
        world_points: np.ndarray,
        world_points_conf: np.ndarray,
        conf_threshold: float,
        max_points: int,
        grid_cell_size: int,
        topk_per_cell: int,
    ) -> Dict[str, object]:
        points = np.asarray(world_points, dtype=np.float32)
        conf = np.asarray(world_points_conf, dtype=np.float32)
        if points.ndim != 3 or points.shape[-1] != 3 or conf.shape != points.shape[:2]:
            return {
                "flat_indices": np.empty((0,), dtype=np.int64),
                "points": np.empty((0, 3), dtype=np.float32),
                "conf": np.empty((0,), dtype=np.float32),
                "shape": tuple(points.shape[:2]) if points.ndim >= 2 else (0, 0),
            }

        valid = np.isfinite(points).all(axis=-1)
        valid &= np.isfinite(conf)
        valid &= conf >= float(conf_threshold)
        if not np.any(valid):
            return {
                "flat_indices": np.empty((0,), dtype=np.int64),
                "points": np.empty((0, 3), dtype=np.float32),
                "conf": np.empty((0,), dtype=np.float32),
                "shape": tuple(points.shape[:2]),
            }

        rows, cols = np.nonzero(valid)
        flat_indices = np.flatnonzero(valid)
        weights = conf[valid].reshape(-1)
        max_points = max(1, int(max_points))
        keep = cls._select_grid_conf_topk_indices(
            weights=weights,
            rows=rows,
            cols=cols,
            max_corr=max_points,
            cell_size=max(1, int(grid_cell_size)),
            topk_per_cell=max(1, int(topk_per_cell)),
        )
        flat_indices = flat_indices[keep]
        weights = weights[keep]

        points_flat = points.reshape(-1, 3)
        return {
            "flat_indices": flat_indices.astype(np.int64, copy=False),
            "points": points_flat[flat_indices].astype(np.float32, copy=False),
            "conf": weights.astype(np.float32, copy=False),
            "shape": tuple(int(v) for v in points.shape[:2]),
            "sampling_mode": "grid_conf_topk",
            "grid_cell_size": int(max(1, int(grid_cell_size))),
            "topk_per_cell": int(max(1, int(topk_per_cell))),
            "max_points": int(max_points),
        }

    @classmethod
    def _collect_compact_frame_correspondences(
        cls,
        *,
        local_points: np.ndarray,
        local_conf: np.ndarray,
        compact_reference: Dict[str, object],
        conf_threshold: float,
        grid_conf_min: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
        local_points_np = np.asarray(local_points, dtype=np.float32)
        local_conf_np = np.asarray(local_conf, dtype=np.float32)
        flat_indices = np.asarray(compact_reference.get("flat_indices"), dtype=np.int64).reshape(-1)
        global_points = np.asarray(compact_reference.get("points"), dtype=np.float32).reshape(-1, 3)
        global_conf = np.asarray(compact_reference.get("conf"), dtype=np.float32).reshape(-1)

        candidate_count = int(flat_indices.shape[0])
        if (
            candidate_count == 0
            or global_points.shape[0] != candidate_count
            or global_conf.shape[0] != candidate_count
            or local_points_np.ndim != 3
            or local_points_np.shape[-1] != 3
            or local_conf_np.shape != local_points_np.shape[:2]
        ):
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                {
                    "sampling_mode": "grid_conf_topk",
                    "candidate_count": candidate_count,
                    "post_conf_min_count": 0,
                    "selected_count": 0,
                    "mean_selected_weight": 0.0,
                },
            )

        local_flat = local_points_np.reshape(-1, 3)
        local_conf_flat = local_conf_np.reshape(-1)
        in_bounds = (flat_indices >= 0) & (flat_indices < local_flat.shape[0])
        if not np.any(in_bounds):
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                {
                    "sampling_mode": "grid_conf_topk",
                    "candidate_count": candidate_count,
                    "post_conf_min_count": 0,
                    "selected_count": 0,
                    "mean_selected_weight": 0.0,
                },
            )

        flat_indices = flat_indices[in_bounds]
        global_points = global_points[in_bounds]
        global_conf = global_conf[in_bounds]
        src = local_flat[flat_indices]
        local_conf_selected = local_conf_flat[flat_indices]
        valid = np.isfinite(src).all(axis=1)
        valid &= np.isfinite(global_points).all(axis=1)
        valid &= np.isfinite(local_conf_selected)
        valid &= np.isfinite(global_conf)
        valid &= local_conf_selected >= float(conf_threshold)
        valid &= global_conf >= float(conf_threshold)
        weights = np.sqrt(local_conf_selected[valid] * global_conf[valid])
        valid &= np.sqrt(np.maximum(local_conf_selected, 0.0) * np.maximum(global_conf, 0.0)) >= float(grid_conf_min)

        if not np.any(valid):
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                {
                    "sampling_mode": "grid_conf_topk",
                    "candidate_count": candidate_count,
                    "post_conf_min_count": 0,
                    "selected_count": 0,
                    "mean_selected_weight": 0.0,
                },
            )

        weights = np.sqrt(local_conf_selected[valid] * global_conf[valid])
        src = src[valid]
        dst = global_points[valid]
        stats = {
            "sampling_mode": "grid_conf_topk",
            "candidate_count": candidate_count,
            "post_conf_min_count": int(dst.shape[0]),
            "selected_count": int(dst.shape[0]),
            "mean_selected_weight": float(np.mean(weights)) if weights.size > 0 else 0.0,
        }
        return (
            src.astype(np.float64, copy=False),
            dst.astype(np.float64, copy=False),
            weights.astype(np.float64, copy=False),
            stats,
        )

    @classmethod
    def _estimate_point_sim3_transform(
        cls,
        subset: SubsetInferenceResult,
        neighbor_frames: List[FrameReconstruction],
        use_scale: bool,
        conf_threshold: float,
        grid_conf_min: float,
        fine_residual_mad_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        src_points = []
        dst_points = []
        weights = []
        per_frame_counts = []
        per_frame_sampling: List[Dict[str, object]] = []
        kept_frame_ids: List[str] = []
        rejected_frame_ids: List[str] = []
        profile: Dict[str, float] = {}
        sampling_totals = {
            "candidate_count": 0,
            "post_conf_min_count": 0,
            "selected_count": 0,
            "selected_weight_sum": 0.0,
        }

        collect_start = time.perf_counter()
        for idx, frame in enumerate(neighbor_frames):
            frame_collect_start = time.perf_counter()
            compact_reference = frame.metadata.get("alignment_compact_reference")
            if not isinstance(compact_reference, dict):
                raise ValueError(
                    f"Frame {frame.frame_id} is missing compact alignment reference data."
                )
            src_frame, dst_frame, w_frame, sampling_stats = cls._collect_compact_frame_correspondences(
                local_points=subset.world_points[idx],
                local_conf=subset.world_points_conf[idx],
                compact_reference=compact_reference,
                conf_threshold=conf_threshold,
                grid_conf_min=grid_conf_min,
            )
            profile["collect_correspondences_sec"] = (
                profile.get("collect_correspondences_sec", 0.0)
                + time.perf_counter() - frame_collect_start
            )
            count = int(src_frame.shape[0])
            per_frame_counts.append(count)
            per_frame_sampling.append({"frame_id": frame.frame_id, **sampling_stats})
            sampling_totals["candidate_count"] += int(sampling_stats.get("candidate_count", 0))
            sampling_totals["post_conf_min_count"] += int(sampling_stats.get("post_conf_min_count", 0))
            sampling_totals["selected_count"] += count
            sampling_totals["selected_weight_sum"] += float(sampling_stats.get("mean_selected_weight", 0.0)) * count
            if count == 0:
                rejected_frame_ids.append(frame.frame_id)
                continue

            kept_frame_ids.append(frame.frame_id)
            src_points.append(src_frame.astype(np.float64, copy=False))
            dst_points.append(dst_frame.astype(np.float64, copy=False))
            weights.append(w_frame.astype(np.float64, copy=False))
        profile["collect_correspondences_total_sec"] = time.perf_counter() - collect_start
        profile["sampling_candidate_count"] = float(sampling_totals["candidate_count"])
        profile["sampling_post_conf_min_count"] = float(sampling_totals["post_conf_min_count"])
        profile["sampling_selected_count"] = float(sampling_totals["selected_count"])
        profile["sampling_mean_selected_weight"] = (
            float(sampling_totals["selected_weight_sum"] / sampling_totals["selected_count"])
            if sampling_totals["selected_count"] > 0
            else 0.0
        )

        if not src_points:
            raise ValueError("No valid correspondences remained for point_sim3 alignment.")

        concat_start = time.perf_counter()
        src_all = np.concatenate(src_points, axis=0)
        dst_all = np.concatenate(dst_points, axis=0)
        w_all = np.concatenate(weights, axis=0)
        w_all = w_all / np.clip(w_all.sum(), 1e-12, None)
        profile["concatenate_correspondences_sec"] = time.perf_counter() - concat_start

        sim3_start = time.perf_counter()
        from third_party_codes.vggt_long_code import (
            NUMBA_SIM3_AVAILABLE,
            robust_weighted_estimate_sim3,
            robust_weighted_estimate_sim3_numba,
        )

        sim3_solver = "numba_irls"
        sim3_fallback_error = None
        if NUMBA_SIM3_AVAILABLE:
            try:
                scale, rotation, translation = robust_weighted_estimate_sim3_numba(
                    src=src_all,
                    tgt=dst_all,
                    init_weights=w_all,
                    use_scale=use_scale,
                )
            except Exception as exc:
                sim3_solver = "numpy_irls_fallback"
                sim3_fallback_error = str(exc)
                scale, rotation, translation = robust_weighted_estimate_sim3(
                    src=src_all,
                    tgt=dst_all,
                    init_weights=w_all,
                    use_scale=use_scale,
                )
        else:
            sim3_solver = "numpy_irls"
            scale, rotation, translation = robust_weighted_estimate_sim3(
                src=src_all,
                tgt=dst_all,
                init_weights=w_all,
                use_scale=use_scale,
            )
        profile["robust_sim3_sec"] = time.perf_counter() - sim3_start
        profile["robust_sim3_used_numba"] = 1.0 if sim3_solver == "numba_irls" else 0.0

        residual_start = time.perf_counter()
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = scale * rotation
        transform[:3, 3] = translation

        predicted = cls._apply_transform_to_xyz(transform, src_all)
        residuals = np.linalg.norm(predicted - dst_all, axis=1)
        inlier_threshold = cls._robust_residual_threshold(
            residuals,
            mad_scale=fine_residual_mad_scale,
            minimum_slack=1e-4,
        )
        inliers = np.flatnonzero(residuals <= inlier_threshold).astype(np.int32)
        profile["residual_inlier_sec"] = time.perf_counter() - residual_start

        diagnostics = {
            "mode": "point_sim3",
            "alignment_reference_mode": "compact",
            "num_shared_frames": int(len(neighbor_frames)),
            "num_correspondences": int(src_all.shape[0]),
            "per_frame_correspondences": per_frame_counts,
            "sampling": {
                "mode": "grid_conf_topk",
                "candidate_count": int(sampling_totals["candidate_count"]),
                "post_conf_min_count": int(sampling_totals["post_conf_min_count"]),
                "selected_count": int(sampling_totals["selected_count"]),
                "mean_selected_weight": (
                    float(sampling_totals["selected_weight_sum"] / sampling_totals["selected_count"])
                    if sampling_totals["selected_count"] > 0
                    else 0.0
                ),
                "per_frame": per_frame_sampling,
            },
            "kept_reference_frames": kept_frame_ids,
            "rejected_reference_frames": rejected_frame_ids,
            "num_inliers": int(len(inliers)),
            "mean_point_residual": float(residuals.mean()),
            "median_point_residual": float(np.median(residuals)),
            "max_point_residual": float(residuals.max()),
            "estimated_scale": float(scale),
            "sim3_solver": sim3_solver,
            "sim3_numba_available": bool(NUMBA_SIM3_AVAILABLE),
            "sim3_fallback_error": sim3_fallback_error,
            "profile": profile,
        }
        return transform.astype(np.float32), inliers, diagnostics

    @staticmethod
    def _apply_transform_to_xyz(transform: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
        transformed = (transform @ xyz_h.T).T
        return transformed[:, :3] / np.clip(transformed[:, 3:4], 1e-12, None)

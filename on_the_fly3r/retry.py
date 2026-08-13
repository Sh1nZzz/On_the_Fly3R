"""Validation-failure reference pruning and retry orchestration."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .types import FrameReconstruction, SubsetInferenceResult
from .utils import _robust_z, _safe_l2_normalize

class RetryMixin:
    def _build_register_ref_pruning_retry_plan(
        self,
        subset: SubsetInferenceResult,
        neighbor_frames: List[FrameReconstruction],
    ) -> tuple[Dict[str, object], List[int]]:
        log: Dict[str, object] = {
            "enabled": bool(self.config.enable_register_ref_pruning_retry),
            "attempted": False,
            "accepted": False,
        }
        if not self.config.enable_register_ref_pruning_retry:
            log["reason"] = "disabled"
            return log, []

        num_refs = len(neighbor_frames)
        min_refs_after_drop = max(
            int(self.config.min_topk_for_alignment),
            int(self.config.register_ref_pruning_min_refs),
        )
        drop_capacity = num_refs - min_refs_after_drop
        if drop_capacity <= 0:
            log.update(
                {
                    "reason": "not_enough_refs_to_drop",
                    "num_refs": int(num_refs),
                    "min_refs_after_drop": int(min_refs_after_drop),
                }
            )
            return log, []

        register_mean = subset.vfm_register_tokens
        if register_mean is None:
            log["reason"] = "missing_vfm_register_tokens"
            if subset.vfm_register_metadata:
                log["vfm_register_metadata"] = dict(subset.vfm_register_metadata)
            return log, []
        register_mean = np.asarray(register_mean, dtype=np.float32)
        if register_mean.ndim != 2 or register_mean.shape[0] != len(subset.frame_ids):
            log.update(
                {
                    "reason": "invalid_vfm_register_shape",
                    "vfm_register_shape": list(register_mean.shape),
                    "num_subset_frames": int(len(subset.frame_ids)),
                }
            )
            return log, []

        new_features = register_mean[num_refs:]
        ref_features = register_mean[:num_refs]
        if new_features.size == 0 or ref_features.size == 0:
            log["reason"] = "missing_ref_or_new_registers"
            return log, []

        new_anchor = _safe_l2_normalize(new_features.mean(axis=0, keepdims=True), axis=1)[0]
        ref_norm = _safe_l2_normalize(ref_features, axis=1)
        similarities = np.clip(ref_norm @ new_anchor, -1.0, 1.0)
        distances = 1.0 - similarities
        distance_z = _robust_z(distances)

        order = np.argsort(-distances)
        candidate_mask = (
            (distance_z >= float(self.config.register_ref_pruning_candidate_z))
            | (similarities <= float(self.config.register_ref_pruning_min_similarity))
        )
        max_drop = min(int(self.config.register_ref_pruning_max_drop_refs), int(drop_capacity))
        candidate_indices = [int(idx) for idx in order if bool(candidate_mask[idx])]
        if candidate_indices:
            drop_indices = candidate_indices[:max_drop]
            selection_mode = "threshold"
        else:
            drop_indices = [int(order[0])] if max_drop > 0 and order.size > 0 else []
            selection_mode = "worst_ref_fallback"

        if not drop_indices:
            log["reason"] = "no_drop_candidate"
            return log, []

        drop_set = set(drop_indices)
        kept_indices = [idx for idx in range(num_refs) if idx not in drop_set]
        ref_scores = []
        for idx, frame in enumerate(neighbor_frames):
            ref_scores.append(
                {
                    "frame_id": frame.frame_id,
                    "similarity_to_new": float(similarities[idx]),
                    "distance_to_new": float(distances[idx]),
                    "distance_z": float(distance_z[idx]),
                    "selected_for_drop": bool(idx in drop_set),
                }
            )

        log.update(
            {
                "planned": True,
                "selection_mode": selection_mode,
                "num_refs_before": int(num_refs),
                "num_refs_after": int(len(kept_indices)),
                "min_refs_after_drop": int(min_refs_after_drop),
                "vfm_register_metadata": dict(subset.vfm_register_metadata),
                "dropped_neighbors": [neighbor_frames[idx].frame_id for idx in drop_indices],
                "kept_neighbors": [neighbor_frames[idx].frame_id for idx in kept_indices],
                "ref_scores": ref_scores,
            }
        )
        return log, kept_indices

    @staticmethod
    def _slice_subset_for_ref_retry(
        subset: SubsetInferenceResult,
        *,
        num_refs: int,
        kept_ref_indices: Sequence[int],
    ) -> SubsetInferenceResult:
        selected_indices = list(kept_ref_indices) + list(range(num_refs, len(subset.frame_ids)))
        register_mean = None
        if subset.vfm_register_tokens is not None:
            register = np.asarray(subset.vfm_register_tokens)
            if register.ndim >= 1 and register.shape[0] == len(subset.frame_ids):
                register_mean = register[selected_indices].astype(np.float32, copy=False)
        return SubsetInferenceResult(
            image_paths=[subset.image_paths[idx] for idx in selected_indices],
            frame_ids=[subset.frame_ids[idx] for idx in selected_indices],
            cam2world=subset.cam2world[selected_indices],
            intrinsic=subset.intrinsic[selected_indices],
            world_points=subset.world_points[selected_indices],
            world_points_conf=subset.world_points_conf[selected_indices],
            images=subset.images[selected_indices],
            vfm_register_tokens=register_mean,
            vfm_register_metadata=dict(subset.vfm_register_metadata),
        )

    @staticmethod
    def _rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
        delta = np.asarray(rotation_a, dtype=np.float64) @ np.asarray(rotation_b, dtype=np.float64).T
        trace = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(trace)))

    @staticmethod
    def _sim3_components(transform: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        transform = np.asarray(transform, dtype=np.float64)
        linear = transform[:3, :3]
        scale = float(np.mean(np.linalg.norm(linear, axis=0)))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        rotation = linear / scale
        U, _, Vt = np.linalg.svd(rotation)
        rotation = U @ Vt
        if np.linalg.det(rotation) < 0:
            U[:, -1] *= -1.0
            rotation = U @ Vt
        return scale, rotation, transform[:3, 3].copy()

    @staticmethod
    def _positive_outlier_scores(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        scores = np.zeros_like(values, dtype=np.float64)
        finite = np.isfinite(values)
        if np.count_nonzero(finite) < 2:
            scores[~finite] = float("inf")
            return scores
        robust = _robust_z(values[finite])
        if np.all(np.abs(robust) < 1e-8):
            std = float(np.std(values[finite]))
            if std > 1e-8:
                robust = (values[finite] - float(np.median(values[finite]))) / std
        scores[finite] = np.maximum(robust, 0.0)
        scores[~finite] = float("inf")
        return scores

    @classmethod
    def _single_ref_transform_consistency(
        cls,
        transforms: Sequence[Optional[np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(transforms)
        translation_dispersion = np.full((n,), np.inf, dtype=np.float64)
        rotation_dispersion = np.full((n,), np.inf, dtype=np.float64)
        log_scale_dispersion = np.full((n,), np.inf, dtype=np.float64)

        components: List[Optional[tuple[float, np.ndarray, np.ndarray]]] = [
            cls._sim3_components(transform) if transform is not None else None
            for transform in transforms
        ]
        for idx, component in enumerate(components):
            if component is None:
                continue
            scale, rotation, translation = component
            trans_values = []
            rot_values = []
            scale_values = []
            for other_idx, other_component in enumerate(components):
                if idx == other_idx or other_component is None:
                    continue
                other_scale, other_rotation, other_translation = other_component
                trans_values.append(float(np.linalg.norm(translation - other_translation)))
                rot_values.append(cls._rotation_angle_deg(rotation, other_rotation))
                scale_values.append(float(abs(np.log(max(scale, 1e-12) / max(other_scale, 1e-12)))))
            if trans_values:
                translation_dispersion[idx] = float(np.median(trans_values))
                rotation_dispersion[idx] = float(np.median(rot_values))
                log_scale_dispersion[idx] = float(np.median(scale_values))
            else:
                translation_dispersion[idx] = 0.0
                rotation_dispersion[idx] = 0.0
                log_scale_dispersion[idx] = 0.0
        return translation_dispersion, rotation_dispersion, log_scale_dispersion

    def _estimate_single_ref_geometry_scores(
        self,
        subset: SubsetInferenceResult,
        neighbor_frames: List[FrameReconstruction],
    ) -> tuple[List[Dict[str, object]], List[Optional[np.ndarray]]]:
        num_refs = len(neighbor_frames)
        transforms: List[Optional[np.ndarray]] = []
        ref_scores: List[Dict[str, object]] = []
        for idx, frame in enumerate(neighbor_frames):
            single_subset = self._slice_subset_for_ref_retry(
                subset,
                num_refs=num_refs,
                kept_ref_indices=[idx],
            )
            try:
                transform, inliers, diagnostics = self._estimate_point_sim3_transform(
                    subset=single_subset,
                    neighbor_frames=[frame],
                    use_scale=self.config.alignment_use_scale,
                    conf_threshold=self.config.alignment_point_conf_threshold,
                    grid_conf_min=self.config.alignment_grid_conf_min,
                    fine_residual_mad_scale=self.config.alignment_fine_residual_mad_scale,
                )
                pred_pose = self._transform_pose(transform, single_subset.cam2world[0])
                translation_residual = float(np.linalg.norm(pred_pose[:3, 3] - frame.cam2world[:3, 3]))
                rotation_residual = self._rotation_angle_deg(pred_pose[:3, :3], frame.cam2world[:3, :3])
                transforms.append(transform.astype(np.float32, copy=False))
                ref_scores.append(
                    {
                        "frame_id": frame.frame_id,
                        "ref_index": int(idx),
                        "single_ref_status": "ok",
                        "num_correspondences": int(diagnostics.get("num_correspondences", 0)),
                        "num_inliers": int(len(inliers)),
                        "median_point_residual": float(diagnostics.get("median_point_residual", float("inf"))),
                        "max_point_residual": float(diagnostics.get("max_point_residual", float("inf"))),
                        "pose_translation_residual": translation_residual,
                        "pose_rotation_residual_deg": rotation_residual,
                        "estimated_scale": float(diagnostics.get("estimated_scale", 1.0)),
                    }
                )
            except Exception as exc:
                transforms.append(None)
                ref_scores.append(
                    {
                        "frame_id": frame.frame_id,
                        "ref_index": int(idx),
                        "single_ref_status": "failed",
                        "error": str(exc),
                        "num_correspondences": 0,
                        "num_inliers": 0,
                        "median_point_residual": float("inf"),
                        "max_point_residual": float("inf"),
                        "pose_translation_residual": float("inf"),
                        "pose_rotation_residual_deg": float("inf"),
                        "estimated_scale": None,
                    }
                )
        return ref_scores, transforms

    def _build_geometry_ref_pruning_retry_plan(
        self,
        subset: SubsetInferenceResult,
        neighbor_frames: List[FrameReconstruction],
    ) -> tuple[Dict[str, object], List[int]]:
        log: Dict[str, object] = {
            "stage": "geometry_existing_subset",
            "attempted": False,
            "accepted": False,
        }
        num_refs = len(neighbor_frames)
        min_refs_after_drop = max(
            int(self.config.min_topk_for_alignment),
            int(self.config.register_ref_pruning_min_refs),
        )
        drop_capacity = num_refs - min_refs_after_drop
        if drop_capacity <= 0:
            log.update(
                {
                    "reason": "not_enough_refs_to_drop",
                    "num_refs": int(num_refs),
                    "min_refs_after_drop": int(min_refs_after_drop),
                }
            )
            return log, []

        ref_scores, transforms = self._estimate_single_ref_geometry_scores(subset, neighbor_frames)
        translation_dispersion, rotation_dispersion, log_scale_dispersion = self._single_ref_transform_consistency(transforms)

        point_values = np.asarray([score["median_point_residual"] for score in ref_scores], dtype=np.float64)
        pose_t_values = np.asarray([score["pose_translation_residual"] for score in ref_scores], dtype=np.float64)
        pose_r_values = np.asarray([score["pose_rotation_residual_deg"] for score in ref_scores], dtype=np.float64)

        outlier_score = (
            self._positive_outlier_scores(translation_dispersion)
            + self._positive_outlier_scores(rotation_dispersion)
            + self._positive_outlier_scores(log_scale_dispersion)
            + self._positive_outlier_scores(point_values)
            + self._positive_outlier_scores(pose_t_values)
            + self._positive_outlier_scores(pose_r_values)
        )

        failed_indices = [idx for idx, transform in enumerate(transforms) if transform is None]
        threshold = float(self.config.register_ref_pruning_candidate_z)
        candidate_indices = [
            int(idx)
            for idx, score in enumerate(outlier_score)
            if idx not in failed_indices and np.isfinite(score) and score >= threshold
        ]
        order = sorted(
            range(num_refs),
            key=lambda idx: (
                idx not in failed_indices,
                -float(outlier_score[idx]) if np.isfinite(outlier_score[idx]) else float("-inf"),
                idx,
            ),
        )
        max_drop = min(int(self.config.register_ref_pruning_max_drop_refs), int(drop_capacity))
        drop_indices: List[int] = []
        selection_mode = "threshold"
        for idx in failed_indices:
            if len(drop_indices) < max_drop:
                drop_indices.append(int(idx))
        for idx in candidate_indices:
            if len(drop_indices) < max_drop and idx not in drop_indices:
                drop_indices.append(int(idx))
        if not drop_indices and max_drop > 0:
            selection_mode = "worst_ref_fallback"
            for idx in order:
                if idx not in drop_indices:
                    drop_indices.append(int(idx))
                    break

        if not drop_indices:
            log["reason"] = "no_drop_candidate"
            return log, []

        drop_set = set(drop_indices)
        kept_indices = [idx for idx in range(num_refs) if idx not in drop_set]
        for idx, score in enumerate(ref_scores):
            score.update(
                {
                    "transform_translation_dispersion": float(translation_dispersion[idx]),
                    "transform_rotation_dispersion_deg": float(rotation_dispersion[idx]),
                    "transform_log_scale_dispersion": float(log_scale_dispersion[idx]),
                    "geometry_outlier_score": float(outlier_score[idx]) if np.isfinite(outlier_score[idx]) else None,
                    "selected_for_drop": bool(idx in drop_set),
                }
            )

        log.update(
            {
                "planned": True,
                "selection_mode": selection_mode,
                "num_refs_before": int(num_refs),
                "num_refs_after": int(len(kept_indices)),
                "min_refs_after_drop": int(min_refs_after_drop),
                "dropped_neighbors": [neighbor_frames[idx].frame_id for idx in drop_indices],
                "kept_neighbors": [neighbor_frames[idx].frame_id for idx in kept_indices],
                "ref_scores": ref_scores,
            }
        )
        return log, kept_indices

    def _attempt_geometry_ref_pruning_retry(
        self,
        *,
        subset: SubsetInferenceResult,
        neighbor_frames: List[FrameReconstruction],
        fresh_frame_ids: Sequence[str],
        initial_validation: Dict[str, object],
    ) -> Optional[Dict[str, object]]:
        retry_log, kept_ref_indices = self._build_geometry_ref_pruning_retry_plan(subset, neighbor_frames)
        retry_log["initial_validation_reason"] = initial_validation.get("reason")
        retry_log["initial_failed_checks"] = list(initial_validation.get("failed_checks", []))
        if not retry_log.get("planned", False):
            return {"retry_log": retry_log}

        retry_log["attempted"] = True
        retry_neighbor_frames = [neighbor_frames[idx] for idx in kept_ref_indices]
        retry_subset = self._slice_subset_for_ref_retry(
            subset,
            num_refs=len(neighbor_frames),
            kept_ref_indices=kept_ref_indices,
        )
        retry_subset.vfm_register_tokens = None

        attempt = self.run_alignment_attempt(
            subset=retry_subset,
            neighbor_frames=retry_neighbor_frames,
            new_frame_label=",".join(fresh_frame_ids),
            validate=True,
        )
        if attempt["alignment_error"] is not None:
            retry_log.update(
                {
                    "retry_status": "alignment_failed",
                    "retry_error": str(attempt["alignment_error"]),
                    "retry_alignment_time_sec": attempt["alignment_time_sec"],
                }
            )
            return {"retry_log": retry_log}
        retry_alignment = attempt["alignment"]
        retry_alignment_time_sec = attempt["alignment_time_sec"]
        retry_validation = attempt["validation"]
        retry_log.update(
            {
                "retry_status": "accepted" if retry_validation["accepted"] else "failed_validation",
                "accepted": bool(retry_validation["accepted"]),
                "retry_validation_reason": retry_validation.get("reason"),
                "retry_failed_checks": list(retry_validation.get("failed_checks", [])),
                "retry_alignment_time_sec": retry_alignment_time_sec,
            }
        )

        if not retry_validation["accepted"]:
            retry_log["retry_validation"] = retry_validation
            return {"retry_log": retry_log}

        return {
            "retry_log": retry_log,
            "subset": retry_subset,
            "neighbor_frames": retry_neighbor_frames,
            "neighbor_ids": [frame.frame_id for frame in retry_neighbor_frames],
            "inference": {"time_sec": 0.0, "reused_initial_subset": True},
            "alignment": retry_alignment,
            "alignment_time_sec": retry_alignment_time_sec,
            "validation": retry_validation,
        }

    def _attempt_register_ref_pruning_retry(
        self,
        *,
        subset: SubsetInferenceResult,
        neighbor_frames: List[FrameReconstruction],
        fresh_paths: Sequence[str],
        fresh_frame_ids: Sequence[str],
        initial_validation: Dict[str, object],
    ) -> Optional[Dict[str, object]]:
        retry_log, kept_ref_indices = self._build_register_ref_pruning_retry_plan(subset, neighbor_frames)
        retry_log["stage"] = "vfm_register_reinference"
        retry_log["initial_validation_reason"] = initial_validation.get("reason")
        retry_log["initial_failed_checks"] = list(initial_validation.get("failed_checks", []))
        if not retry_log.get("planned", False):
            return {"retry_log": retry_log}

        retry_log["attempted"] = True
        retry_neighbor_frames = [neighbor_frames[idx] for idx in kept_ref_indices]
        retry_subset_paths = [frame.image_path for frame in retry_neighbor_frames] + list(fresh_paths)

        try:
            retry_subset, retry_inference_stats = self._measure_stage(
                lambda: self.runner.infer_subset(retry_subset_paths),
                device_type=self.config.inference_device,
            )
            if retry_subset.profile:
                retry_inference_stats["profile"] = dict(retry_subset.profile)
        except Exception as exc:
            retry_log.update(
                {
                    "retry_status": "inference_failed",
                    "retry_error": str(exc),
                }
            )
            return {"retry_log": retry_log}
        retry_subset.vfm_register_tokens = None

        attempt = self.run_alignment_attempt(
            subset=retry_subset,
            neighbor_frames=retry_neighbor_frames,
            new_frame_label=",".join(fresh_frame_ids),
            validate=True,
        )
        if attempt["alignment_error"] is not None:
            retry_log.update(
                {
                    "retry_status": "alignment_failed",
                    "retry_error": str(attempt["alignment_error"]),
                    "retry_alignment_time_sec": attempt["alignment_time_sec"],
                    "retry_inference": retry_inference_stats,
                }
            )
            return {"retry_log": retry_log}
        retry_alignment = attempt["alignment"]
        retry_alignment_time_sec = attempt["alignment_time_sec"]
        retry_validation = attempt["validation"]
        retry_log.update(
            {
                "retry_status": "accepted" if retry_validation["accepted"] else "failed_validation",
                "accepted": bool(retry_validation["accepted"]),
                "retry_validation_reason": retry_validation.get("reason"),
                "retry_failed_checks": list(retry_validation.get("failed_checks", [])),
                "retry_inference": retry_inference_stats,
                "retry_alignment_time_sec": retry_alignment_time_sec,
            }
        )

        if not retry_validation["accepted"]:
            retry_log["retry_validation"] = retry_validation
            return {"retry_log": retry_log}

        return {
            "retry_log": retry_log,
            "subset": retry_subset,
            "neighbor_frames": retry_neighbor_frames,
            "neighbor_ids": [frame.frame_id for frame in retry_neighbor_frames],
            "inference": retry_inference_stats,
            "alignment": retry_alignment,
            "alignment_time_sec": retry_alignment_time_sec,
            "validation": retry_validation,
        }

    def _attempt_two_stage_ref_pruning_retry(
        self,
        *,
        subset: SubsetInferenceResult,
        neighbor_frames: List[FrameReconstruction],
        fresh_paths: Sequence[str],
        fresh_frame_ids: Sequence[str],
        initial_validation: Dict[str, object],
    ) -> Optional[Dict[str, object]]:
        combined_log: Dict[str, object] = {
            "enabled": bool(self.config.enable_register_ref_pruning_retry),
            "strategy": "geometry_existing_subset_then_vfm_register_reinference",
            "attempted": False,
            "accepted": False,
            "initial_validation_reason": initial_validation.get("reason"),
            "initial_failed_checks": list(initial_validation.get("failed_checks", [])),
            "stages": [],
        }
        if not self.config.enable_register_ref_pruning_retry:
            combined_log["reason"] = "disabled"
            return {"retry_log": combined_log}

        geometry_result = self._attempt_geometry_ref_pruning_retry(
            subset=subset,
            neighbor_frames=neighbor_frames,
            fresh_frame_ids=fresh_frame_ids,
            initial_validation=initial_validation,
        )
        geometry_log = geometry_result.get("retry_log") if geometry_result is not None else None
        if geometry_log is not None:
            combined_log["attempted"] = bool(combined_log["attempted"] or geometry_log.get("attempted", False))
            combined_log["stages"].append(geometry_log)
        if geometry_result is not None and geometry_result.get("validation", {}).get("accepted", False):
            combined_log.update(
                {
                    "accepted": True,
                    "accepted_stage": "geometry_existing_subset",
                    "retry_status": "accepted",
                    "dropped_neighbors": geometry_log.get("dropped_neighbors", []) if geometry_log else [],
                    "kept_neighbors": geometry_log.get("kept_neighbors", []) if geometry_log else [],
                }
            )
            return {**geometry_result, "retry_log": combined_log}

        register_result = self._attempt_register_ref_pruning_retry(
            subset=subset,
            neighbor_frames=neighbor_frames,
            fresh_paths=fresh_paths,
            fresh_frame_ids=fresh_frame_ids,
            initial_validation=initial_validation,
        )
        register_log = register_result.get("retry_log") if register_result is not None else None
        if register_log is not None:
            combined_log["attempted"] = bool(combined_log["attempted"] or register_log.get("attempted", False))
            combined_log["stages"].append(register_log)
        if register_result is not None and register_result.get("validation", {}).get("accepted", False):
            combined_log.update(
                {
                    "accepted": True,
                    "accepted_stage": "vfm_register_reinference",
                    "retry_status": "accepted",
                    "dropped_neighbors": register_log.get("dropped_neighbors", []) if register_log else [],
                    "kept_neighbors": register_log.get("kept_neighbors", []) if register_log else [],
                }
            )
            return {**register_result, "retry_log": combined_log}

        last_stage_log = combined_log["stages"][-1] if combined_log["stages"] else {}
        combined_log.update(
            {
                "retry_status": last_stage_log.get("retry_status", "not_accepted"),
                "retry_validation_reason": last_stage_log.get("retry_validation_reason"),
                "retry_failed_checks": last_stage_log.get("retry_failed_checks", []),
            }
        )
        return {"retry_log": combined_log}

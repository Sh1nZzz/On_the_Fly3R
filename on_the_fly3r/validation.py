"""Post-alignment pose validation and dataset spacing estimation."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .types import AlignmentResult, FrameReconstruction, SubsetInferenceResult

class ValidationMixin:
    def _get_validation_translation_threshold(self) -> float:
        if self.validation_translation_threshold is not None:
            return float(self.validation_translation_threshold)
        return 0.1

    def _estimate_dataset_spacing_reference(
        self,
        frames: Sequence[FrameReconstruction],
    ) -> Optional[float]:
        if not frames:
            return None

        max_frames = max(1, int(self.config.validation_spacing_estimation_max_frames))
        samples_per_frame = max(
            128,
            int(self.config.validation_spacing_estimation_samples // max_frames),
        )

        collected: List[np.ndarray] = []
        for frame in list(frames)[:max_frames]:
            if frame.world_points is None or frame.world_points_conf is None:
                continue
            spacing = self._collect_local_spacing_samples(
                world_points=frame.world_points,
                world_points_conf=frame.world_points_conf,
                conf_threshold=self.config.fusion_conf_threshold,
                max_samples=samples_per_frame,
            )
            if spacing.size > 0:
                collected.append(spacing)

        if not collected:
            return None

        distances = np.concatenate(collected, axis=0).astype(np.float32, copy=False)
        valid = np.isfinite(distances) & (distances > 0)
        if not np.any(valid):
            return None
        return float(np.median(distances[valid]))

    def _estimate_dataset_spacing_reference_from_dense_batch(
        self,
        world_points: np.ndarray,
        world_points_conf: np.ndarray,
    ) -> Optional[float]:
        points_batch = np.asarray(world_points)
        conf_batch = np.asarray(world_points_conf)
        if points_batch.ndim != 4 or points_batch.shape[-1] != 3 or conf_batch.shape != points_batch.shape[:3]:
            return None

        max_frames = max(1, int(self.config.validation_spacing_estimation_max_frames))
        samples_per_frame = max(
            128,
            int(self.config.validation_spacing_estimation_samples // max_frames),
        )
        collected: List[np.ndarray] = []
        for idx in range(min(max_frames, points_batch.shape[0])):
            spacing = self._collect_local_spacing_samples(
                world_points=points_batch[idx],
                world_points_conf=conf_batch[idx],
                conf_threshold=self.config.fusion_conf_threshold,
                max_samples=samples_per_frame,
            )
            if spacing.size > 0:
                collected.append(spacing)
        if not collected:
            return None
        distances = np.concatenate(collected, axis=0).astype(np.float32, copy=False)
        valid = np.isfinite(distances) & (distances > 0)
        if not np.any(valid):
            return None
        return float(np.median(distances[valid]))

    def _validate_alignment_result(
        self,
        subset: SubsetInferenceResult,
        neighbor_frames: List[FrameReconstruction],
        alignment: AlignmentResult,
    ) -> Dict[str, object]:
        diagnostics = dict(alignment.diagnostics)
        num_refs = len(neighbor_frames)
        if num_refs <= 0:
            return {
                "accepted": False,
                "reason": "no_reference_frames",
                "failed_checks": ["no_reference_frames"],
                "metrics": {},
            }

        pred_poses = np.stack(
            [
                self._transform_pose(alignment.transform_local_to_global, subset.cam2world[idx])
                for idx in range(num_refs)
            ],
            axis=0,
        ).astype(np.float64)
        gt_poses = np.stack([frame.cam2world for frame in neighbor_frames], axis=0).astype(np.float64)

        translation_residuals = np.linalg.norm(pred_poses[:, :3, 3] - gt_poses[:, :3, 3], axis=1)
        rotation_residuals_deg = []
        for idx in range(num_refs):
            dR = pred_poses[idx, :3, :3] @ gt_poses[idx, :3, :3].T
            trace = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
            rotation_residuals_deg.append(float(np.degrees(np.arccos(trace))))
        rotation_residuals_deg = np.asarray(rotation_residuals_deg, dtype=np.float64)

        estimated_scale = diagnostics.get("estimated_scale")
        estimated_scale = float(estimated_scale) if estimated_scale is not None else None

        metrics = {
            "mode": diagnostics.get("mode", "point_sim3"),
            "num_shared_frames": int(num_refs),
            "translation_residuals": translation_residuals.astype(np.float32).tolist(),
            "rotation_residuals_deg": rotation_residuals_deg.astype(np.float32).tolist(),
            "median_translation_residual": float(np.median(translation_residuals)),
            "max_translation_residual": float(np.max(translation_residuals)),
            "mean_translation_residual": float(np.mean(translation_residuals)),
            "median_rotation_residual_deg": float(np.median(rotation_residuals_deg)),
            "max_rotation_residual_deg": float(np.max(rotation_residuals_deg)),
            "mean_rotation_residual_deg": float(np.mean(rotation_residuals_deg)),
            "estimated_scale": estimated_scale,
        }

        translation_threshold = self._get_validation_translation_threshold()

        failed_checks: List[str] = []
        if metrics["median_translation_residual"] > translation_threshold:
            failed_checks.append("median_translation_residual")
        if metrics["max_translation_residual"] > translation_threshold:
            failed_checks.append("max_translation_residual")
        if metrics["median_rotation_residual_deg"] > self.config.validation_median_rotation_deg_threshold:
            failed_checks.append("median_rotation_residual_deg")
        if metrics["max_rotation_residual_deg"] > self.config.validation_max_rotation_deg_threshold:
            failed_checks.append("max_rotation_residual_deg")
        if estimated_scale is not None and (
            estimated_scale < self.config.validation_scale_min
            or estimated_scale > self.config.validation_scale_max
        ):
            failed_checks.append("estimated_scale")
        accepted = len(failed_checks) == 0
        return {
            "accepted": accepted,
            "reason": "ok" if accepted else failed_checks[0],
            "failed_checks": failed_checks,
            "metrics": {
                **metrics,
                "translation_threshold": float(translation_threshold),
            },
        }

    @staticmethod
    def _collect_local_spacing_samples(
        world_points: np.ndarray,
        world_points_conf: np.ndarray,
        conf_threshold: float,
        max_samples: int,
    ) -> np.ndarray:
        points = np.asarray(world_points, dtype=np.float32)
        conf = np.asarray(world_points_conf, dtype=np.float32)
        if points.ndim != 3 or points.shape[-1] != 3 or conf.ndim != 2:
            return np.empty((0,), dtype=np.float32)

        valid = np.isfinite(points).all(axis=-1)
        valid &= np.isfinite(conf)
        valid &= conf >= conf_threshold

        distances: List[np.ndarray] = []
        if points.shape[1] > 1:
            valid_x = valid[:, :-1] & valid[:, 1:]
            if np.any(valid_x):
                dx = np.linalg.norm(points[:, 1:, :] - points[:, :-1, :], axis=-1)
                distances.append(dx[valid_x].astype(np.float32, copy=False))

        if points.shape[0] > 1:
            valid_y = valid[:-1, :] & valid[1:, :]
            if np.any(valid_y):
                dy = np.linalg.norm(points[1:, :, :] - points[:-1, :, :], axis=-1)
                distances.append(dy[valid_y].astype(np.float32, copy=False))

        if not distances:
            return np.empty((0,), dtype=np.float32)

        samples = np.concatenate(distances, axis=0).astype(np.float32, copy=False)
        if samples.shape[0] > max_samples:
            sample_indices = np.random.choice(samples.shape[0], size=max_samples, replace=False)
            samples = samples[sample_indices]
        return samples

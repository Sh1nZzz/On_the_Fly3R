"""Robust Sim(3) estimators adapted from VGGT-Long.

Source: https://github.com/DengKaiCQ/VGGT-Long/blob/main/loop_utils/sim3utils.py
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration dependency
    njit = None


NUMBA_SIM3_AVAILABLE = njit is not None


def weighted_estimate_sim3(source_points, target_points, weights, use_scale):
    """Estimate a weighted Sim(3) transform from paired ``Nx3`` points."""
    total_weight = np.sum(weights)
    if total_weight < 1e-6:
        raise ValueError("Total weight too small for meaningful estimation")

    normalized_weights = weights / total_weight
    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)
    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    if not use_scale:
        scale = 1.0
    else:
        scale_src = np.sqrt(np.sum(normalized_weights * np.sum(src_centered**2, axis=1)))
        scale_tgt = np.sqrt(np.sum(normalized_weights * np.sum(tgt_centered**2, axis=1)))
        scale = scale_tgt / scale_src

    weighted_src = (scale * src_centered) * np.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights)[:, None]
    covariance = weighted_src.T @ weighted_tgt
    left_vectors, _, rotation_transpose = np.linalg.svd(covariance)
    rotation = rotation_transpose.T @ left_vectors.T

    if np.linalg.det(rotation) < 0:
        rotation_transpose[2, :] *= -1
        rotation = rotation_transpose.T @ left_vectors.T

    translation = mu_tgt - scale * rotation @ mu_src
    return scale, rotation, translation


def huber_loss(residuals, delta):
    absolute = np.abs(residuals)
    return np.where(absolute <= delta, 0.5 * residuals**2, delta * (absolute - 0.5 * delta))


def robust_weighted_estimate_sim3(
    src,
    tgt,
    init_weights,
    delta=0.1,
    max_iters=20,
    tol=1e-9,
    use_scale=True,
):
    """Estimate Sim(3) with confidence weights and iterative Huber reweighting."""
    scale, rotation, translation = weighted_estimate_sim3(src, tgt, init_weights, use_scale=use_scale)
    previous_error = float("inf")

    for _ in range(max_iters):
        transformed = scale * (src @ rotation.T) + translation
        residuals = np.linalg.norm(tgt - transformed, axis=1)

        absolute = np.abs(residuals)
        huber_weights = np.ones_like(residuals)
        large_residuals = absolute > delta
        huber_weights[large_residuals] = delta / absolute[large_residuals]

        combined_weights = init_weights * huber_weights
        combined_weights /= np.sum(combined_weights) + 1e-12
        scale_new, rotation_new, translation_new = weighted_estimate_sim3(
            src,
            tgt,
            combined_weights,
            use_scale=use_scale,
        )

        parameter_change = abs(scale_new - scale) + np.linalg.norm(translation_new - translation)
        rotation_angle = np.arccos(
            min(1.0, max(-1.0, (np.trace(rotation_new @ rotation.T) - 1) / 2))
        )
        current_error = np.sum(huber_loss(residuals, delta) * init_weights)
        if (parameter_change < tol and rotation_angle < np.radians(0.1)) or (
            abs(previous_error - current_error) < tol * previous_error
        ):
            break

        scale, rotation, translation = scale_new, rotation_new, translation_new
        previous_error = current_error

    return scale, rotation, translation


if NUMBA_SIM3_AVAILABLE:

    @njit(cache=True)
    def _weighted_estimate_sim3_moments_numba(source_points, target_points, weights, use_scale):
        total_weight = np.float32(0.0)
        for i in range(weights.shape[0]):
            total_weight += weights[i]
        if total_weight < np.float32(1e-6):
            return (
                np.float32(-1.0),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros((3, 3), dtype=np.float32),
            )

        mu_src = np.zeros(3, dtype=np.float32)
        mu_tgt = np.zeros(3, dtype=np.float32)
        for i in range(source_points.shape[0]):
            weight = weights[i] / total_weight
            for j in range(3):
                mu_src[j] += weight * source_points[i, j]
                mu_tgt[j] += weight * target_points[i, j]

        src_var = np.float32(0.0)
        tgt_var = np.float32(0.0)
        for i in range(source_points.shape[0]):
            weight = weights[i] / total_weight
            src_sq = np.float32(0.0)
            tgt_sq = np.float32(0.0)
            for j in range(3):
                src_diff = source_points[i, j] - mu_src[j]
                tgt_diff = target_points[i, j] - mu_tgt[j]
                src_sq += src_diff * src_diff
                tgt_sq += tgt_diff * tgt_diff
            src_var += weight * src_sq
            tgt_var += weight * tgt_sq

        if use_scale:
            if src_var <= np.float32(1e-12):
                return (
                    np.float32(-1.0),
                    np.zeros(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.zeros((3, 3), dtype=np.float32),
                )
            scale = np.sqrt(tgt_var / src_var)
        else:
            scale = np.float32(1.0)

        covariance = np.zeros((3, 3), dtype=np.float32)
        for i in range(source_points.shape[0]):
            weight = weights[i] / total_weight
            for row in range(3):
                src_diff = source_points[i, row] - mu_src[row]
                for col in range(3):
                    tgt_diff = target_points[i, col] - mu_tgt[col]
                    covariance[row, col] += weight * scale * src_diff * tgt_diff

        return scale, mu_src, mu_tgt, covariance


    @njit(cache=True)
    def _residuals_numba(src, tgt, scale, rotation, translation):
        residuals = np.empty(src.shape[0], dtype=np.float32)
        for i in range(src.shape[0]):
            x = (
                scale
                * (
                    rotation[0, 0] * src[i, 0]
                    + rotation[0, 1] * src[i, 1]
                    + rotation[0, 2] * src[i, 2]
                )
                + translation[0]
            )
            y = (
                scale
                * (
                    rotation[1, 0] * src[i, 0]
                    + rotation[1, 1] * src[i, 1]
                    + rotation[1, 2] * src[i, 2]
                )
                + translation[1]
            )
            z = (
                scale
                * (
                    rotation[2, 0] * src[i, 0]
                    + rotation[2, 1] * src[i, 1]
                    + rotation[2, 2] * src[i, 2]
                )
                + translation[2]
            )
            dx = tgt[i, 0] - x
            dy = tgt[i, 1] - y
            dz = tgt[i, 2] - z
            residuals[i] = np.sqrt(dx * dx + dy * dy + dz * dz)
        return residuals


    @njit(cache=True)
    def _huber_weights_numba(residuals, delta):
        weights = np.ones(residuals.shape[0], dtype=np.float32)
        for i in range(residuals.shape[0]):
            residual = residuals[i]
            if residual > delta:
                weights[i] = delta / residual
        return weights


    @njit(cache=True)
    def _weighted_huber_error_numba(residuals, init_weights, delta):
        total = np.float32(0.0)
        for i in range(residuals.shape[0]):
            residual = residuals[i]
            if residual <= delta:
                loss = np.float32(0.5) * residual * residual
            else:
                loss = delta * (residual - np.float32(0.5) * delta)
            total += loss * init_weights[i]
        return total


    def weighted_estimate_sim3_numba(source_points, target_points, weights, use_scale=True):
        source_points = np.ascontiguousarray(source_points, dtype=np.float32)
        target_points = np.ascontiguousarray(target_points, dtype=np.float32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)

        scale, mu_src, mu_tgt, covariance = _weighted_estimate_sim3_moments_numba(
            source_points,
            target_points,
            weights,
            bool(use_scale),
        )
        if scale < 0:
            raise ValueError("Total weight too small for meaningful estimation")

        left_vectors, _, rotation_transpose = np.linalg.svd(covariance.astype(np.float32))
        rotation = rotation_transpose.T @ left_vectors.T
        if np.linalg.det(rotation) < 0:
            rotation_transpose[2, :] *= -1
            rotation = rotation_transpose.T @ left_vectors.T

        translation = mu_tgt - scale * rotation @ mu_src
        return float(scale), rotation.astype(np.float64), translation.astype(np.float64)


    def robust_weighted_estimate_sim3_numba(
        src,
        tgt,
        init_weights,
        delta=0.1,
        max_iters=20,
        tol=1e-9,
        use_scale=True,
    ):
        src = np.ascontiguousarray(src, dtype=np.float32)
        tgt = np.ascontiguousarray(tgt, dtype=np.float32)
        init_weights = np.ascontiguousarray(init_weights, dtype=np.float32)
        delta = np.float32(delta)

        scale, rotation, translation = weighted_estimate_sim3_numba(
            src,
            tgt,
            init_weights,
            use_scale=use_scale,
        )
        previous_error = float("inf")

        for _ in range(max_iters):
            residuals = _residuals_numba(
                src,
                tgt,
                np.float32(scale),
                rotation.astype(np.float32),
                translation.astype(np.float32),
            )
            huber_weights = _huber_weights_numba(residuals, delta)
            combined_weights = init_weights * huber_weights
            combined_weights /= np.sum(combined_weights) + np.float32(1e-12)

            scale_new, rotation_new, translation_new = weighted_estimate_sim3_numba(
                src,
                tgt,
                combined_weights,
                use_scale=use_scale,
            )

            parameter_change = abs(scale_new - scale) + np.linalg.norm(translation_new - translation)
            rotation_argument = (np.trace(rotation_new @ rotation.T) - 1.0) / 2.0
            rotation_angle = np.arccos(min(1.0, max(-1.0, rotation_argument)))
            current_error = float(_weighted_huber_error_numba(residuals, init_weights, delta))

            if (parameter_change < tol and rotation_angle < np.radians(0.1)) or (
                np.isfinite(previous_error)
                and abs(previous_error - current_error) < tol * abs(previous_error)
            ):
                break

            scale, rotation, translation = scale_new, rotation_new, translation_new
            previous_error = current_error

        return scale, rotation, translation

else:

    def weighted_estimate_sim3_numba(source_points, target_points, weights, use_scale=True):
        raise ImportError("numba is not available; use weighted_estimate_sim3 instead.")


    def robust_weighted_estimate_sim3_numba(
        src,
        tgt,
        init_weights,
        delta=0.1,
        max_iters=20,
        tol=1e-9,
        use_scale=True,
    ):
        raise ImportError("numba is not available; use robust_weighted_estimate_sim3 instead.")

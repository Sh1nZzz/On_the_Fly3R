import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ColmapImagePose:
    image_id: int
    camera_id: int
    image_name: str
    world2cam: np.ndarray
    cam2world: np.ndarray

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate exported camera_poses.npz against COLMAP images.txt poses.")
    parser.add_argument("--pred_npz", type=str, required=True, help="Path to camera_poses.npz exported by run_on_the_fly3r.py.")
    parser.add_argument("--gt_colmap", type=str, required=True, help="Path to COLMAP images.txt style ground-truth poses.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write eval_metrics.json and error CSV files. Defaults to pred_npz parent.")
    parser.add_argument("--align", type=str, default="sim3", choices=["sim3", "se3", "none"], help="Global alignment applied to predicted cam2world poses before evaluation.")
    parser.add_argument("--match_by", type=str, default="auto", choices=["auto", "name", "stem"], help="How to match COLMAP image names to predicted image paths.")
    parser.add_argument("--eval_frame_filter_npz", type=str, default=None, help="Optional camera_poses.npz whose valid=True frames define the common evaluation subset.")
    parser.add_argument("--relative_delta", type=int, default=1, help="Frame interval used for relative pose metrics.")
    parser.add_argument("--relative_pair_mode", type=str, default="delta", choices=["delta", "all", "sampled"], help="Relative pair selection: fixed delta, all valid GT-order pairs, or sampled all-pair subset.")
    parser.add_argument("--max_relative_pairs", type=int, default=0, help="Maximum all-pair relative pairs to evaluate when --relative_pair_mode sampled, or an optional cap for all mode. <=0 disables the cap.")
    parser.add_argument("--relative_pair_seed", type=int, default=0, help="Random seed for sampled relative pairs.")
    parser.add_argument("--rotation_thresholds_deg", type=str, default="1,3,5,10", help="Comma-separated thresholds for RRA.")
    parser.add_argument("--translation_angle_thresholds_deg", type=str, default=None, help="Comma-separated translation direction angle thresholds in degrees for relative RTA. Defaults to 1,3,5.")
    parser.add_argument("--translation_thresholds", type=str, default=None, help="Deprecated alias for --translation_angle_thresholds_deg.")
    return parser


def _parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _normalize_path_text(value: str) -> str:
    return str(value).replace("\\", "/").strip()


def _basename(value: str) -> str:
    return _normalize_path_text(value).split("/")[-1]


def _stem(value: str) -> str:
    return Path(_basename(value)).stem


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(qvec)
    if norm <= 0:
        raise ValueError("Encountered a zero-norm COLMAP quaternion.")
    w, x, y, z = qvec / norm
    return np.array(
        [
            [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * w * z, 2.0 * x * z + 2.0 * w * y],
            [2.0 * x * y + 2.0 * w * z, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * w * x],
            [2.0 * x * z - 2.0 * w * y, 2.0 * y * z + 2.0 * w * x, 1.0 - 2.0 * x * x - 2.0 * y * y],
        ],
        dtype=np.float64,
    )


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def _translation_direction_angle_deg(
    pred_translation: np.ndarray,
    gt_translation: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Optional[float]:
    pred = np.asarray(pred_translation, dtype=np.float64).reshape(3)
    gt = np.asarray(gt_translation, dtype=np.float64).reshape(3)
    pred_norm = float(np.linalg.norm(pred))
    gt_norm = float(np.linalg.norm(gt))
    if pred_norm <= eps or gt_norm <= eps:
        return None
    value = float(np.dot(pred, gt) / (pred_norm * gt_norm))
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def _read_colmap_images_txt(path: str) -> list[ColmapImagePose]:
    records: list[ColmapImagePose] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.split()
            if len(tokens) < 10:
                continue

            try:
                image_id = int(tokens[0])
                qvec = np.asarray([float(value) for value in tokens[1:5]], dtype=np.float64)
                tvec = np.asarray([float(value) for value in tokens[5:8]], dtype=np.float64)
                camera_id = int(tokens[8])
            except ValueError:
                continue

            image_name = " ".join(tokens[9:])
            world2cam = np.eye(4, dtype=np.float64)
            world2cam[:3, :3] = _qvec_to_rotmat(qvec)
            world2cam[:3, 3] = tvec
            cam2world = np.linalg.inv(world2cam)
            records.append(
                ColmapImagePose(
                    image_id=image_id,
                    camera_id=camera_id,
                    image_name=image_name,
                    world2cam=world2cam,
                    cam2world=cam2world,
                )
            )

    if not records:
        raise ValueError(f"No COLMAP image pose records were parsed from {path}.")
    return records


def _read_pred_npz(path: str) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    required = {"frame_ids", "image_paths", "cam2world"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"{path} is missing required arrays: {missing}")

    cam2world = np.asarray(data["cam2world"], dtype=np.float64)
    if cam2world.ndim != 3 or cam2world.shape[1:] != (4, 4):
        raise ValueError("pred cam2world must have shape (N,4,4).")

    valid = np.asarray(data["valid"], dtype=bool) if "valid" in data.files else np.isfinite(cam2world).all(axis=(1, 2))
    valid = valid & np.isfinite(cam2world).all(axis=(1, 2))

    return {
        "frame_ids": [str(item) for item in data["frame_ids"].tolist()],
        "image_paths": [str(item) for item in data["image_paths"].tolist()],
        "cam2world": cam2world,
        "valid": valid,
    }


def _read_valid_frame_filter(path: Optional[str]) -> Optional[dict[str, set[str]]]:
    if path is None:
        return None
    data = np.load(path, allow_pickle=True)
    required = {"frame_ids", "image_paths", "valid"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"{path} is missing required filter arrays: {missing}")

    frame_ids = [str(item) for item in data["frame_ids"].tolist()]
    image_paths = [str(item) for item in data["image_paths"].tolist()]
    valid = np.asarray(data["valid"], dtype=bool)

    names: set[str] = set()
    stems: set[str] = set()
    for frame_id, image_path, is_valid in zip(frame_ids, image_paths, valid.tolist()):
        if not is_valid:
            continue
        names.add(_basename(image_path))
        stems.add(_stem(image_path))
        stems.add(str(frame_id))

    return {"names": names, "stems": stems}


def _passes_frame_filter(gt_image_name: str, frame_filter: Optional[dict[str, set[str]]]) -> bool:
    if frame_filter is None:
        return True
    return _basename(gt_image_name) in frame_filter["names"] or _stem(gt_image_name) in frame_filter["stems"]


def _add_to_lookup(lookup: dict[str, list[int]], key: str, idx: int) -> None:
    if not key:
        return
    indices = lookup.setdefault(key, [])
    if idx not in indices:
        indices.append(idx)


def _build_pred_lookups(image_paths: list[str], frame_ids: list[str]) -> dict[str, dict[str, list[int]]]:
    by_name: dict[str, list[int]] = {}
    by_stem: dict[str, list[int]] = {}
    for idx, (image_path, frame_id) in enumerate(zip(image_paths, frame_ids)):
        _add_to_lookup(by_name, _basename(image_path), idx)
        _add_to_lookup(by_stem, _stem(image_path), idx)
        _add_to_lookup(by_stem, str(frame_id), idx)
    return {"name": by_name, "stem": by_stem}


def _lookup_unique(lookup: dict[str, list[int]], key: str) -> Optional[int]:
    matches = lookup.get(key, [])
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple predicted frames match key '{key}'. Use a less ambiguous --match_by mode or unique image names.")
    return int(matches[0])


def _match_pred_index(
    gt_image_name: str,
    lookups: dict[str, dict[str, list[int]]],
    match_by: str,
) -> Optional[int]:
    if match_by in {"auto", "name"}:
        idx = _lookup_unique(lookups["name"], _basename(gt_image_name))
        if idx is not None or match_by == "name":
            return idx

    return _lookup_unique(lookups["stem"], _stem(gt_image_name))


def _umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("Umeyama inputs must both have shape (N,3).")
    if src.shape[0] < 3:
        raise ValueError("At least 3 valid pose correspondences are required for trajectory alignment.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    covariance = (dst_centered.T @ src_centered) / src.shape[0]

    U, singular_values, Vt = np.linalg.svd(covariance)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1.0
    rotation = U @ S @ Vt

    if with_scale:
        variance = np.mean(np.sum(src_centered * src_centered, axis=1))
        scale = float(np.sum(singular_values * np.diag(S)) / max(variance, 1e-12))
    else:
        scale = 1.0

    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation, translation


def _apply_similarity_to_cam2world(pose: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation @ pose[:3, :3]
    out[:3, 3] = scale * (rotation @ pose[:3, 3]) + translation
    return out


def _summarize(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"rmse": None, "mean": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _accuracy(values: list[float], thresholds: list[float], unit: str) -> dict[str, Optional[float]]:
    if not values:
        return {f"@{threshold:g}{unit}": None for threshold in thresholds}
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"@{threshold:g}{unit}": float(np.mean(arr <= threshold))
        for threshold in thresholds
    }


def _upper_triangle_pair_from_linear(index: int, num_items: int) -> tuple[int, int]:
    # Pairs are ordered as (0,1), (0,2), ..., (1,2), ...
    if num_items < 2:
        raise ValueError("At least two items are required for upper-triangle pair indexing.")
    total = num_items * (num_items - 1) // 2
    if index < 0 or index >= total:
        raise IndexError(f"Pair index {index} is out of range for {num_items} items.")
    value = (2 * num_items - 1) ** 2 - 8 * index
    row = int(np.floor(((2 * num_items - 1) - np.sqrt(value)) * 0.5))
    row_start = row * (2 * num_items - row - 1) // 2
    col = row + 1 + (index - row_start)
    return row, int(col)


def _sample_upper_triangle_pairs(
    candidate_gt_indices: list[int],
    *,
    max_pairs: int,
    seed: int,
) -> list[tuple[int, int]]:
    num_items = len(candidate_gt_indices)
    total_pairs = num_items * (num_items - 1) // 2
    if max_pairs <= 0 or max_pairs >= total_pairs:
        return [
            (candidate_gt_indices[src], candidate_gt_indices[dst])
            for src in range(num_items)
            for dst in range(src + 1, num_items)
        ]
    rng = np.random.default_rng(int(seed))
    sampled = np.sort(rng.choice(total_pairs, size=int(max_pairs), replace=False))
    pairs: list[tuple[int, int]] = []
    for linear_idx in sampled.tolist():
        src_pos, dst_pos = _upper_triangle_pair_from_linear(int(linear_idx), num_items)
        pairs.append((candidate_gt_indices[src_pos], candidate_gt_indices[dst_pos]))
    return pairs


def _build_relative_pairs(
    *,
    candidate_gt_indices: list[int],
    num_gt_records: int,
    delta: int,
    mode: str,
    max_pairs: int,
    seed: int,
) -> tuple[list[tuple[int, int]], dict[str, object]]:
    candidate_gt_indices = sorted(int(idx) for idx in candidate_gt_indices)
    if mode == "delta":
        candidate_gt_set = set(candidate_gt_indices)
        all_pairs = [
            (src_idx, src_idx + delta)
            for src_idx in range(max(0, num_gt_records - delta))
            if src_idx in candidate_gt_set and (src_idx + delta) in candidate_gt_set
        ]
        return all_pairs, {
            "mode": mode,
            "delta": int(delta),
            "num_candidate_pairs_before_sampling": int(len(all_pairs)),
            "num_candidate_pairs_after_sampling": int(len(all_pairs)),
            "max_relative_pairs": int(max_pairs),
            "sampled": False,
        }

    num_items = len(candidate_gt_indices)
    all_pair_count = num_items * (num_items - 1) // 2
    should_sample = mode == "sampled" or (max_pairs > 0 and max_pairs < all_pair_count)
    if should_sample:
        if max_pairs <= 0:
            raise ValueError("--relative_pair_mode sampled requires --max_relative_pairs > 0.")
        pairs = _sample_upper_triangle_pairs(candidate_gt_indices, max_pairs=max_pairs, seed=seed)
    else:
        pairs = [
            (candidate_gt_indices[src], candidate_gt_indices[dst])
            for src in range(num_items)
            for dst in range(src + 1, num_items)
        ]
    return pairs, {
        "mode": mode,
        "delta": None,
        "num_candidate_pairs_before_sampling": int(all_pair_count),
        "num_candidate_pairs_after_sampling": int(len(pairs)),
        "max_relative_pairs": int(max_pairs),
        "sampled": bool(should_sample),
        "seed": int(seed) if should_sample else None,
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    gt_records = _read_colmap_images_txt(args.gt_colmap)
    pred = _read_pred_npz(args.pred_npz)
    frame_filter = _read_valid_frame_filter(getattr(args, "eval_frame_filter_npz", None))
    frame_ids = pred["frame_ids"]
    image_paths = pred["image_paths"]
    pred_cam2world = pred["cam2world"]
    pred_valid = pred["valid"]
    lookups = _build_pred_lookups(image_paths=image_paths, frame_ids=frame_ids)

    matched_pred_indices: list[Optional[int]] = []
    candidate_gt_indices: list[int] = []
    filtered_out_gt_image_names: list[str] = []
    missing_gt_image_names: list[str] = []
    invalid_pred_image_names: list[str] = []
    valid_gt_indices: list[int] = []
    for gt_idx, record in enumerate(gt_records):
        if not _passes_frame_filter(record.image_name, frame_filter):
            matched_pred_indices.append(None)
            filtered_out_gt_image_names.append(record.image_name)
            continue

        candidate_gt_indices.append(gt_idx)
        pred_idx = _match_pred_index(record.image_name, lookups, args.match_by)
        matched_pred_indices.append(pred_idx)
        if pred_idx is None:
            missing_gt_image_names.append(record.image_name)
            continue
        if not bool(pred_valid[pred_idx]):
            invalid_pred_image_names.append(record.image_name)
            continue
        valid_gt_indices.append(gt_idx)

    if not valid_gt_indices:
        raise ValueError("No valid predicted poses matched the COLMAP ground truth.")

    gt_centers = np.stack([gt_records[idx].cam2world[:3, 3] for idx in valid_gt_indices], axis=0)
    pred_centers = np.stack([pred_cam2world[matched_pred_indices[idx]][:3, 3] for idx in valid_gt_indices], axis=0)

    if args.align == "none":
        scale = 1.0
        align_rotation = np.eye(3, dtype=np.float64)
        align_translation = np.zeros(3, dtype=np.float64)
    else:
        scale, align_rotation, align_translation = _umeyama_alignment(
            src=pred_centers,
            dst=gt_centers,
            with_scale=args.align == "sim3",
        )

    aligned_pred_by_gt_idx: dict[int, np.ndarray] = {}
    for gt_idx in valid_gt_indices:
        pred_idx = matched_pred_indices[gt_idx]
        aligned_pred_by_gt_idx[gt_idx] = _apply_similarity_to_cam2world(
            pred_cam2world[pred_idx],
            scale=scale,
            rotation=align_rotation,
            translation=align_translation,
        )

    abs_rows: list[dict[str, object]] = []
    abs_translation_errors: list[float] = []
    abs_rotation_errors_deg: list[float] = []
    for gt_idx in valid_gt_indices:
        gt_pose = gt_records[gt_idx].cam2world
        pred_pose = aligned_pred_by_gt_idx[gt_idx]
        translation_error = float(np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3]))
        rotation_error = _rotation_angle_deg(pred_pose[:3, :3] @ gt_pose[:3, :3].T)
        abs_translation_errors.append(translation_error)
        abs_rotation_errors_deg.append(rotation_error)
        abs_rows.append(
            {
                "gt_index": gt_idx,
                "image_name": gt_records[gt_idx].image_name,
                "translation_error": translation_error,
                "rotation_error_deg": rotation_error,
            }
        )

    delta = max(1, int(args.relative_delta))
    relative_pair_mode = getattr(args, "relative_pair_mode", "delta")
    max_relative_pairs = max(0, int(getattr(args, "max_relative_pairs", 0)))
    relative_pair_seed = int(getattr(args, "relative_pair_seed", 0))
    rel_rows: list[dict[str, object]] = []
    rel_translation_angle_errors_deg: list[float] = []
    rel_translation_norm_errors: list[float] = []
    rel_rotation_errors_deg: list[float] = []
    skipped_relative_pairs: list[dict[str, object]] = []
    relative_pairs, relative_pair_selection = _build_relative_pairs(
        candidate_gt_indices=candidate_gt_indices,
        num_gt_records=len(gt_records),
        delta=delta,
        mode=relative_pair_mode,
        max_pairs=max_relative_pairs,
        seed=relative_pair_seed,
    )

    for src_idx, dst_idx in relative_pairs:
        if src_idx not in aligned_pred_by_gt_idx or dst_idx not in aligned_pred_by_gt_idx:
            skipped_relative_pairs.append(
                {
                    "src_index": src_idx,
                    "dst_index": dst_idx,
                    "src_image_name": gt_records[src_idx].image_name,
                    "dst_image_name": gt_records[dst_idx].image_name,
                    "reason": "missing_or_invalid_pred_pose",
                }
            )
            continue

        gt_relative = np.linalg.inv(gt_records[src_idx].cam2world) @ gt_records[dst_idx].cam2world
        pred_relative = np.linalg.inv(aligned_pred_by_gt_idx[src_idx]) @ aligned_pred_by_gt_idx[dst_idx]
        relative_error = np.linalg.inv(gt_relative) @ pred_relative
        translation_norm_error = float(np.linalg.norm(relative_error[:3, 3]))
        translation_angle_error_deg = _translation_direction_angle_deg(
            pred_relative[:3, 3],
            gt_relative[:3, 3],
        )
        rotation_error = _rotation_angle_deg(relative_error[:3, :3])
        if translation_angle_error_deg is not None:
            rel_translation_angle_errors_deg.append(translation_angle_error_deg)
        rel_translation_norm_errors.append(translation_norm_error)
        rel_rotation_errors_deg.append(rotation_error)
        rel_rows.append(
            {
                "src_index": src_idx,
                "dst_index": dst_idx,
                "src_image_name": gt_records[src_idx].image_name,
                "dst_image_name": gt_records[dst_idx].image_name,
                "pair_index_gap": int(dst_idx - src_idx),
                "gt_baseline": float(np.linalg.norm(gt_relative[:3, 3])),
                "pred_baseline": float(np.linalg.norm(pred_relative[:3, 3])),
                "translation_angle_error_deg": translation_angle_error_deg,
                "translation_norm_error": translation_norm_error,
                "rotation_error_deg": rotation_error,
            }
        )

    rotation_thresholds = _parse_thresholds(args.rotation_thresholds_deg)
    translation_angle_thresholds_text = (
        getattr(args, "translation_angle_thresholds_deg", None)
        or getattr(args, "translation_thresholds", None)
        or "1,3,5"
    )
    translation_angle_thresholds = _parse_thresholds(translation_angle_thresholds_text)

    absolute_translation_stats = _summarize(abs_translation_errors)
    absolute_rotation_stats = _summarize(abs_rotation_errors_deg)
    relative_translation_angle_stats = _summarize(rel_translation_angle_errors_deg)
    relative_translation_norm_stats = _summarize(rel_translation_norm_errors)
    relative_rotation_stats = _summarize(rel_rotation_errors_deg)

    metrics = {
        "inputs": {
            "pred_npz": str(args.pred_npz),
            "gt_colmap": str(args.gt_colmap),
            "match_by": args.match_by,
            "eval_frame_filter_npz": getattr(args, "eval_frame_filter_npz", None),
            "relative_pair_mode": relative_pair_mode,
            "relative_delta": int(delta),
            "max_relative_pairs": int(max_relative_pairs),
        },
        "alignment": {
            "mode": args.align,
            "scale": float(scale),
            "rotation": align_rotation.tolist(),
            "translation": align_translation.tolist(),
        },
        "coverage": {
            "num_gt_frames": int(len(gt_records)),
            "num_candidate_gt_frames": int(len(candidate_gt_indices)),
            "num_filtered_out_gt_frames": int(len(filtered_out_gt_image_names)),
            "num_pred_frames": int(len(image_paths)),
            "num_matched_frames": int(sum(idx is not None for idx in matched_pred_indices)),
            "num_valid_matched_frames": int(len(valid_gt_indices)),
            "num_missing_gt_frames": int(len(missing_gt_image_names)),
            "num_invalid_pred_frames": int(len(invalid_pred_image_names)),
            "valid_frame_coverage": _safe_ratio(len(valid_gt_indices), len(candidate_gt_indices)),
            "valid_frame_coverage_over_all_gt": _safe_ratio(len(valid_gt_indices), len(gt_records)),
            "filtered_out_gt_image_names": filtered_out_gt_image_names,
            "missing_gt_image_names": missing_gt_image_names,
            "invalid_pred_image_names": invalid_pred_image_names,
        },
        "absolute": {
            "num_evaluated_frames": int(len(abs_rows)),
            "translation": absolute_translation_stats,
            "rotation_deg": absolute_rotation_stats,
            "ATE_RMSE": absolute_translation_stats["rmse"],
            "ARE_RMSE_deg": absolute_rotation_stats["rmse"],
        },
        "relative": {
            "pair_selection": relative_pair_selection,
            "delta": int(delta) if relative_pair_mode == "delta" else None,
            "translation_metric": "direction_angle_deg",
            "num_gt_pairs_total": int(relative_pair_selection["num_candidate_pairs_before_sampling"]),
            "num_candidate_pairs_after_sampling": int(relative_pair_selection["num_candidate_pairs_after_sampling"]),
            "num_evaluated_pairs": int(len(rel_rows)),
            "num_skipped_pairs_missing_or_invalid": int(len(skipped_relative_pairs)),
            "evaluated_pair_coverage": _safe_ratio(len(rel_rows), int(relative_pair_selection["num_candidate_pairs_after_sampling"])),
            "num_translation_angle_valid_pairs": int(len(rel_translation_angle_errors_deg)),
            "translation": relative_translation_angle_stats,
            "translation_angle_deg": relative_translation_angle_stats,
            "translation_norm": relative_translation_norm_stats,
            "rotation_deg": relative_rotation_stats,
            "RTE_RMSE": relative_translation_angle_stats["rmse"],
            "RTE_RMSE_deg": relative_translation_angle_stats["rmse"],
            "RRE_RMSE_deg": relative_rotation_stats["rmse"],
            "RTA": _accuracy(rel_translation_angle_errors_deg, translation_angle_thresholds, "deg"),
            "RRA": _accuracy(rel_rotation_errors_deg, rotation_thresholds, "deg"),
        },
    }

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.pred_npz).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "eval_metrics.json"
    metrics["outputs"] = {
        "metrics": str(metrics_path),
        "absolute_pose_errors": str(output_dir / "absolute_pose_errors.csv"),
        "relative_pose_errors": str(output_dir / "relative_pose_errors.csv"),
        "relative_skipped_pairs": str(output_dir / "relative_skipped_pairs.csv"),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    _write_csv(
        output_dir / "absolute_pose_errors.csv",
        abs_rows,
        ["gt_index", "image_name", "translation_error", "rotation_error_deg"],
    )
    _write_csv(
        output_dir / "relative_pose_errors.csv",
        rel_rows,
        [
            "src_index",
            "dst_index",
            "src_image_name",
            "dst_image_name",
            "pair_index_gap",
            "gt_baseline",
            "pred_baseline",
            "translation_angle_error_deg",
            "translation_norm_error",
            "rotation_error_deg",
        ],
    )
    _write_csv(
        output_dir / "relative_skipped_pairs.csv",
        skipped_relative_pairs,
        ["src_index", "dst_index", "src_image_name", "dst_image_name", "reason"],
    )
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate(args)
    output_path = metrics["outputs"]["metrics"]
    print(f"Saved pose evaluation metrics to {output_path}")
    print(
        "Coverage: "
        f"{metrics['coverage']['num_valid_matched_frames']}/"
        f"{metrics['coverage']['num_gt_frames']} valid frames, "
        f"{metrics['relative']['num_evaluated_pairs']}/"
        f"{metrics['relative']['num_gt_pairs_total']} relative pairs"
    )


if __name__ == "__main__":
    main()

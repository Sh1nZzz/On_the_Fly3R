"""Raw and Numba voxel-deduplicated point-cloud export."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import time
from typing import Dict, List, Sequence

import numpy as np

_PLY_VERTEX_DTYPE = [
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("red", "u1"),
    ("green", "u1"),
    ("blue", "u1"),
]

_NUMBA_PACKED_VOXEL_KEEP = None


def _binary_ply_header(vertex_count: int) -> bytes:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {int(vertex_count)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    return header.encode("ascii")


def _pack_ply_vertices(points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    vertex_data = np.empty(len(points), dtype=_PLY_VERTEX_DTYPE)
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    return vertex_data


def _write_binary_ply_streaming(
    point_chunks: Sequence[np.ndarray],
    color_chunks: Sequence[np.ndarray],
    output_path: str,
) -> int:
    if len(point_chunks) != len(color_chunks):
        raise ValueError(
            "Cannot stream PLY because point/color chunk counts differ: "
            f"{len(point_chunks)} vs {len(color_chunks)}."
        )

    valid_counts: List[int] = []
    total_vertices = 0
    for points in point_chunks:
        points_np = np.asarray(points, dtype=np.float32)
        if points_np.size == 0:
            valid_count = 0
        else:
            valid_count = int(np.count_nonzero(np.isfinite(points_np).all(axis=1)))
        valid_counts.append(valid_count)
        total_vertices += valid_count

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(_binary_ply_header(total_vertices))
        for points, colors, valid_count in zip(point_chunks, color_chunks, valid_counts):
            if valid_count <= 0:
                continue
            points_np = np.asarray(points, dtype=np.float32)
            colors_np = np.asarray(colors, dtype=np.uint8)
            if points_np.shape[0] != colors_np.shape[0]:
                raise ValueError(
                    "Cannot stream PLY because point/color chunk lengths differ: "
                    f"{points_np.shape[0]} vs {colors_np.shape[0]}."
                )
            valid = np.isfinite(points_np).all(axis=1)
            if not np.any(valid):
                continue
            vertex_data = _pack_ply_vertices(points_np[valid], colors_np[valid])
            vertex_data.tofile(f)
    return int(total_vertices)


def _numba_packed_voxel_keep_impl(
    keys: np.ndarray,
    key_min: np.ndarray,
    shift_x: int,
    shift_y: int,
    occupied,
) -> np.ndarray:
    keep = np.zeros((keys.shape[0],), dtype=np.bool_)
    for idx in range(keys.shape[0]):
        x = keys[idx, 0] - key_min[0]
        y = keys[idx, 1] - key_min[1]
        z = keys[idx, 2] - key_min[2]
        packed = (
            (np.uint64(x) << shift_x)
            | (np.uint64(y) << shift_y)
            | np.uint64(z)
        )
        if packed in occupied:
            continue
        occupied[packed] = np.uint8(1)
        keep[idx] = True
    return keep


def _get_numba_packed_voxel_keep():
    global _NUMBA_PACKED_VOXEL_KEEP
    if _NUMBA_PACKED_VOXEL_KEEP is None:
        try:
            from numba import njit
        except ImportError as exc:
            raise ImportError(
                "Numba final PLY dedup backend requested, but `numba` is not installed. "
                "Install it in the active environment with `python -m pip install numba`."
            ) from exc
        _NUMBA_PACKED_VOXEL_KEEP = njit(_numba_packed_voxel_keep_impl)
    return _NUMBA_PACKED_VOXEL_KEEP


def _scan_voxel_key_bounds(
    point_chunks: Sequence[np.ndarray],
    voxel_size: float,
) -> Dict[str, object]:
    raw_vertices = 0
    key_min = None
    key_max = None
    scan_start = time.perf_counter()
    for points in point_chunks:
        points_np = np.asarray(points, dtype=np.float32)
        if points_np.size == 0:
            continue
        valid = np.isfinite(points_np).all(axis=1)
        valid_count = int(np.count_nonzero(valid))
        raw_vertices += valid_count
        if valid_count == 0:
            continue
        keys = np.floor(points_np[valid] / voxel_size).astype(np.int64, copy=False)
        chunk_min = np.min(keys, axis=0)
        chunk_max = np.max(keys, axis=0)
        if key_min is None:
            key_min = chunk_min
            key_max = chunk_max
        else:
            key_min = np.minimum(key_min, chunk_min)
            key_max = np.maximum(key_max, chunk_max)

    return {
        "raw_vertices": int(raw_vertices),
        "key_min": None if key_min is None else np.asarray(key_min, dtype=np.int64),
        "key_max": None if key_max is None else np.asarray(key_max, dtype=np.int64),
        "scan_time_sec": time.perf_counter() - scan_start,
    }


def _packed_voxel_key_layout(key_min: np.ndarray, key_max: np.ndarray) -> Dict[str, object]:
    ranges = np.asarray(key_max, dtype=np.int64) - np.asarray(key_min, dtype=np.int64) + 1
    if ranges.shape != (3,) or np.any(ranges <= 0):
        raise ValueError(
            "Cannot use numba packed final PLY dedup because voxel key bounds are invalid: "
            f"min={key_min}, max={key_max}."
        )
    bits = [max(1, int(value - 1).bit_length()) for value in ranges.tolist()]
    total_bits = int(sum(bits))
    if total_bits > 64:
        raise ValueError(
            "Cannot use numba packed final PLY dedup without key collisions because the "
            f"voxel key range needs {total_bits} bits ({bits}). "
            "Increase `--final_ply_dedup_voxel_multiplier`."
        )
    return {
        "bits": bits,
        "total_bits": total_bits,
        "shift_x": int(bits[1] + bits[2]),
        "shift_y": int(bits[2]),
    }


def _write_binary_ply_voxel_dedup_streaming_numba(
    point_chunks: Sequence[np.ndarray],
    color_chunks: Sequence[np.ndarray],
    output_path: str,
    *,
    voxel_size: float,
) -> Dict[str, object]:
    try:
        from numba import types
        from numba.typed import Dict as NumbaDict
    except ImportError as exc:
        raise ImportError(
            "Numba final PLY dedup backend requested, but `numba` is not installed. "
            "Install it in the active environment with `python -m pip install numba`."
        ) from exc

    scan = _scan_voxel_key_bounds(point_chunks, voxel_size)
    raw_vertices = int(scan["raw_vertices"])
    key_min = scan["key_min"]
    key_max = scan["key_max"]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw_vertices == 0 or key_min is None or key_max is None:
        with path.open("wb") as output:
            output.write(_binary_ply_header(0))
        return {
            "raw_vertex_count": int(raw_vertices),
            "exported_vertex_count": 0,
            "voxel_size": float(voxel_size),
            "dedup_ratio": 0.0,
            "dedup_backend": "numba_packed",
            "occupied_voxel_count": 0,
            "packed_key_bits": [0, 0, 0],
            "packed_key_total_bits": 0,
            "key_scan_time_sec": float(scan["scan_time_sec"]),
        }

    layout = _packed_voxel_key_layout(key_min, key_max)
    packed_keep = _get_numba_packed_voxel_keep()
    occupied = NumbaDict.empty(key_type=types.uint64, value_type=types.uint8)

    exported_vertices = 0
    payload_path = None
    filter_time_sec = 0.0
    payload_write_time_sec = 0.0
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            delete=False,
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".payload",
        ) as payload:
            payload_path = Path(payload.name)
            for points, colors in zip(point_chunks, color_chunks):
                points_np = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
                colors_np = np.asarray(colors, dtype=np.uint8)
                if points_np.size == 0:
                    continue
                if points_np.shape[0] != colors_np.shape[0]:
                    raise ValueError(
                        "Cannot stream dedup PLY because point/color chunk lengths differ: "
                        f"{points_np.shape[0]} vs {colors_np.shape[0]}."
                    )
                valid = np.isfinite(points_np).all(axis=1)
                if not np.any(valid):
                    continue
                valid_indices = np.flatnonzero(valid)
                keys = np.ascontiguousarray(
                    np.floor(points_np[valid_indices] / voxel_size).astype(np.int64, copy=False)
                )
                filter_start = time.perf_counter()
                keep_valid = packed_keep(
                    keys,
                    key_min,
                    int(layout["shift_x"]),
                    int(layout["shift_y"]),
                    occupied,
                )
                filter_time_sec += time.perf_counter() - filter_start
                if not np.any(keep_valid):
                    continue
                keep = valid_indices[keep_valid]
                write_start = time.perf_counter()
                vertex_data = _pack_ply_vertices(points_np[keep], colors_np[keep])
                vertex_data.tofile(payload)
                payload_write_time_sec += time.perf_counter() - write_start
                exported_vertices += int(keep.shape[0])

        with path.open("wb") as output:
            output.write(_binary_ply_header(exported_vertices))
            if payload_path is not None:
                with payload_path.open("rb") as payload_in:
                    shutil.copyfileobj(payload_in, output, length=1024 * 1024)
    finally:
        if payload_path is not None:
            try:
                payload_path.unlink()
            except FileNotFoundError:
                pass

    return {
        "raw_vertex_count": int(raw_vertices),
        "exported_vertex_count": int(exported_vertices),
        "voxel_size": float(voxel_size),
        "dedup_ratio": (
            float(exported_vertices) / float(raw_vertices)
            if raw_vertices > 0
            else 0.0
        ),
        "dedup_backend": "numba_packed",
        "occupied_voxel_count": int(len(occupied)),
        "packed_key_bits": [int(v) for v in layout["bits"]],
        "packed_key_total_bits": int(layout["total_bits"]),
        "packed_key_min": [int(v) for v in key_min.tolist()],
        "packed_key_max": [int(v) for v in key_max.tolist()],
        "key_scan_time_sec": float(scan["scan_time_sec"]),
        "numba_filter_time_sec": float(filter_time_sec),
        "payload_write_time_sec": float(payload_write_time_sec),
    }


def _write_binary_ply_voxel_dedup_streaming(
    point_chunks: Sequence[np.ndarray],
    color_chunks: Sequence[np.ndarray],
    output_path: str,
    *,
    voxel_size: float,
) -> Dict[str, object]:
    if len(point_chunks) != len(color_chunks):
        raise ValueError(
            "Cannot stream dedup PLY because point/color chunk counts differ: "
            f"{len(point_chunks)} vs {len(color_chunks)}."
        )
    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive and finite, got {voxel_size}.")

    return _write_binary_ply_voxel_dedup_streaming_numba(
        point_chunks,
        color_chunks,
        output_path,
        voxel_size=voxel_size,
    )


class PointCloudMixin:
    def export_global_point_cloud(self) -> Dict[str, np.ndarray]:
        return self.state.export_global_point_cloud()

    def _count_global_valid_vertices(self) -> int:
        total = 0
        for points in self.state.global_points:
            points_np = np.asarray(points, dtype=np.float32)
            if points_np.size == 0:
                continue
            total += int(np.count_nonzero(np.isfinite(points_np).all(axis=1)))
        return int(total)

    def _final_ply_voxel_size(self, multiplier: float) -> float:
        spacing = self.dataset_spacing_reference
        if spacing is None or not np.isfinite(float(spacing)) or float(spacing) <= 0:
            if self._count_global_valid_vertices() == 0:
                return 1.0
            raise ValueError(
                "Cannot export dedup PLY because dataset_spacing_reference is unavailable. "
                "Use raw PLY export or disable final PLY dedup."
            )
        voxel_size = float(spacing) * max(0.0, float(multiplier))
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError(
                "Cannot export dedup PLY because final dedup voxel size is invalid: "
                f"spacing={spacing}, multiplier={multiplier}."
            )
        return float(voxel_size)

    def export_global_ply(
        self,
        output_path: str,
        *,
        mode: str = "raw",
        final_dedup_voxel_multiplier: float = 1.0,
    ) -> Dict[str, object]:
        export_start = time.perf_counter()
        mode = str(mode).strip().lower()
        if mode == "raw":
            exported_vertices = _write_binary_ply_streaming(
                self.state.global_points,
                self.state.global_colors,
                output_path,
            )
            return {
                "mode": "raw",
                "path": str(output_path),
                "raw_vertex_count": int(exported_vertices),
                "exported_vertex_count": int(exported_vertices),
                "voxel_size": None,
                "dedup_ratio": 1.0 if exported_vertices > 0 else 0.0,
                "export_time_sec": time.perf_counter() - export_start,
            }
        if mode != "dedup":
            raise ValueError(f"Unsupported PLY export mode for a single file: {mode}")

        voxel_size = self._final_ply_voxel_size(final_dedup_voxel_multiplier)
        stats = _write_binary_ply_voxel_dedup_streaming(
            self.state.global_points,
            self.state.global_colors,
            output_path,
            voxel_size=voxel_size,
        )
        stats.update(
            {
                "mode": "dedup",
                "path": str(output_path),
                "voxel_multiplier": float(final_dedup_voxel_multiplier),
                "dedup_backend": "numba",
                "export_time_sec": time.perf_counter() - export_start,
            }
        )
        return stats

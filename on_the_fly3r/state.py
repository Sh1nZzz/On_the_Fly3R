from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .types import FrameReconstruction

class ReconstructionState:
    """
    CPU-side state for the progressive reconstruction. Per-frame dense maps
    are discarded; only compact alignment references and fused chunks remain.
    """

    def __init__(self) -> None:
        self.frames: Dict[str, FrameReconstruction] = {}
        self.frame_order: List[str] = []
        self.frame_order_index: Dict[str, int] = {}
        self.global_points: List[np.ndarray] = []
        self.global_colors: List[np.ndarray] = []
        self.global_weights: List[np.ndarray] = []

    def has_frame(self, frame_id: str) -> bool:
        return frame_id in self.frames

    def add_frame(self, frame: FrameReconstruction) -> None:
        if frame.optimized_cam2world is None:
            frame.optimized_cam2world = np.asarray(frame.cam2world, dtype=np.float32).copy()
        if frame.frame_id not in self.frames:
            self.frame_order_index[frame.frame_id] = len(self.frame_order)
            self.frame_order.append(frame.frame_id)
        self.frames[frame.frame_id] = frame

    def get_frame(self, frame_id: str) -> FrameReconstruction:
        return self.frames[frame_id]

    def get_frames(self, frame_ids: Sequence[str]) -> List[FrameReconstruction]:
        return [self.frames[frame_id] for frame_id in frame_ids]

    def ordered_frame_ids(self) -> List[str]:
        return list(self.frame_order)

    def frame_count(self) -> int:
        return len(self.frame_order)

    def get_frame_order_index(self, frame_id: str) -> Optional[int]:
        return self.frame_order_index.get(frame_id)

    def get_map_pose(self, frame_id: str) -> np.ndarray:
        return self.frames[frame_id].cam2world

    def get_optimized_pose(self, frame_id: str) -> np.ndarray:
        frame = self.frames[frame_id]
        if frame.optimized_cam2world is None:
            return frame.cam2world
        return frame.optimized_cam2world

    def set_optimized_pose(self, frame_id: str, pose: np.ndarray) -> None:
        self.frames[frame_id].optimized_cam2world = np.asarray(pose, dtype=np.float32)

    def append_fused_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        if points.size == 0:
            return None
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.uint8)
        if weights is None:
            weights = np.ones((points.shape[0],), dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32).reshape(-1)
            if weights.shape[0] != points.shape[0]:
                weights = np.ones((points.shape[0],), dtype=np.float32)
        self.global_points.append(points)
        self.global_colors.append(colors)
        self.global_weights.append(weights)
        return len(self.global_points) - 1

    def export_global_point_cloud(self) -> Dict[str, np.ndarray]:
        if not self.global_points:
            return {
                "points": np.empty((0, 3), dtype=np.float32),
                "colors": np.empty((0, 3), dtype=np.uint8),
            }
        points = np.concatenate(self.global_points, axis=0)
        colors = np.concatenate(self.global_colors, axis=0)
        valid = np.isfinite(points).all(axis=1)
        return {
            "points": points[valid],
            "colors": colors[valid],
        }

    def export_camera_poses(
        self,
        image_paths: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        if image_paths is None:
            frames = list(self.frames.values())
            image_paths = [frame.image_path for frame in frames]
        else:
            image_paths = list(image_paths)

        frame_ids = [Path(path).stem for path in image_paths]
        cam2world_map = np.full((len(frame_ids), 4, 4), np.nan, dtype=np.float32)
        cam2world_optimized = np.full((len(frame_ids), 4, 4), np.nan, dtype=np.float32)
        intrinsic = np.full((len(frame_ids), 3, 3), np.nan, dtype=np.float32)
        valid = np.zeros((len(frame_ids),), dtype=bool)

        for idx, frame_id in enumerate(frame_ids):
            frame = self.frames.get(frame_id)
            if frame is None:
                continue
            cam2world_map[idx] = frame.cam2world.astype(np.float32, copy=False)
            optimized_pose = (
                frame.optimized_cam2world
                if frame.optimized_cam2world is not None
                else frame.cam2world
            )
            cam2world_optimized[idx] = np.asarray(optimized_pose, dtype=np.float32)
            intrinsic[idx] = frame.intrinsic.astype(np.float32, copy=False)
            valid[idx] = True

        return {
            "frame_ids": np.asarray(frame_ids),
            "image_paths": np.asarray([str(path) for path in image_paths]),
            "cam2world": cam2world_optimized,
            "cam2world_optimized": cam2world_optimized,
            "cam2world_map": cam2world_map,
            "intrinsic": intrinsic,
            "valid": valid,
        }

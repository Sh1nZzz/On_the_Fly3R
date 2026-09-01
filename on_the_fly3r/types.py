from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

@dataclass
class FrameReconstruction:
    frame_id: str
    image_path: str
    cam2world: np.ndarray
    intrinsic: np.ndarray
    world_points: Optional[np.ndarray]
    world_points_conf: Optional[np.ndarray]
    image: Optional[np.ndarray]
    optimized_cam2world: Optional[np.ndarray] = None
    retrieval_vector: Optional[np.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)

@dataclass
class SubsetInferenceResult:
    image_paths: List[str]
    frame_ids: List[str]
    cam2world: np.ndarray
    intrinsic: np.ndarray
    world_points: np.ndarray
    world_points_conf: np.ndarray
    images: np.ndarray
    vfm_register_tokens: Optional[np.ndarray] = None
    vfm_register_metadata: Dict[str, object] = field(default_factory=dict)
    profile: Dict[str, float] = field(default_factory=dict)

@dataclass
class AlignmentResult:
    transform_local_to_global: np.ndarray
    inlier_indices: np.ndarray
    diagnostics: Dict[str, object] = field(default_factory=dict)

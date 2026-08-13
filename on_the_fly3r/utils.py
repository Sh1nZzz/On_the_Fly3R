"""Small shared helpers for devices, seeds, numeric normalization, and image discovery."""

import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _resolve_runtime_device(requested: str) -> str:
    requested = str(requested).strip()
    if requested.lower().startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return requested
    if requested.lower() == "cpu":
        return "cpu"
    return requested

def _safe_l2_normalize(array: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    denom = np.linalg.norm(array, axis=axis, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return array / denom

def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-8:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad

def _normalize_feature_vector(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature = feature / norm
    return feature

def iter_image_paths(image_root: str, suffixes: Iterable[str] = (".jpg", ".png", ".jpeg")) -> List[str]:
    root = Path(image_root)
    suffixes = {suffix.lower() for suffix in suffixes}
    return [
        str(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in suffixes
    ]

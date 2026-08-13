import math
import threading
from collections import OrderedDict
from typing import Any, Dict, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tvf


def _canonical_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower().replace("_", "-")
    aliases = {
        "vggt": "VGGT",
        "pi3": "Pi3",
        "pi3x": "Pi3x",
        "mapanything": "MapAnything",
        "map-anything": "MapAnything",
        "vggtomega": "VGGTOmega",
        "vggt-omega": "VGGTOmega",
    }
    return aliases.get(normalized, str(model_name))


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.nelement() * value.element_size())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return int(sum(_tensor_bytes(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return int(sum(_tensor_bytes(item) for item in value))
    return 0


def _clone_cached(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_cached(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cached(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cached(item) for item in value)
    return value


def _load_rgb_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        return img.convert("RGB")


def _pi3_target_size_from_image(first_img: Image.Image, pixel_limit: int) -> tuple[int, int]:
    width_orig, height_orig = first_img.size
    scale = math.sqrt(pixel_limit / (width_orig * height_orig)) if width_orig * height_orig > 0 else 1.0
    width_target, height_target = width_orig * scale, height_orig * scale
    k, m = round(width_target / 14), round(height_target / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / max(m, 1) > width_target / max(height_target, 1e-6):
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def _vggt_preprocess_one(image_path: str, mode: str) -> torch.Tensor:
    target_size = 518
    img = _load_rgb_image(image_path)
    width, height = img.size

    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
    else:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    tensor = tvf.ToTensor()(img)

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        tensor = tensor[:, start_y : start_y + target_size, :]

    if mode == "pad":
        h_padding = target_size - tensor.shape[1]
        w_padding = target_size - tensor.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = torch.nn.functional.pad(
                tensor,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
    return tensor.to(dtype=torch.float32)


def _pad_tensors_to_common_size(tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    shapes = {(int(tensor.shape[1]), int(tensor.shape[2])) for tensor in tensors}
    if len(shapes) <= 1:
        return list(tensors)
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)
    padded = []
    for tensor in tensors:
        h_padding = max_height - tensor.shape[1]
        w_padding = max_width - tensor.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = torch.nn.functional.pad(
                tensor,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        padded.append(tensor)
    return padded


def _omega_crop_to_supported_aspect_ratio(
    image: Image.Image,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> Image.Image:
    width, height = image.size
    aspect_ratio = height / max(width, 1)
    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return image.crop((left, 0, left + crop_width, height))
    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return image.crop((0, top, width, top + crop_height))
    return image


def _omega_balanced_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> tuple[int, int]:
    token_number = (image_resolution // patch_size) ** 2
    w_patches = math.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(round(w_patches)))
    h_patches = max(1, int(round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def _omega_max_size_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> tuple[int, int]:
    if aspect_ratio >= 1.0:
        height = image_resolution
        width = max(patch_size, int(round((image_resolution / aspect_ratio) / patch_size)) * patch_size)
    else:
        width = image_resolution
        height = max(patch_size, int(round((image_resolution * aspect_ratio) / patch_size)) * patch_size)
    return height, width


def _omega_preprocess_one(
    image_path: str,
    *,
    mode: str,
    image_resolution: int,
    patch_size: int,
) -> torch.Tensor:
    image = _omega_crop_to_supported_aspect_ratio(_load_rgb_image(image_path))
    width, height = image.size
    aspect_ratio = height / max(width, 1)
    if mode == "balanced":
        target_h, target_w = _omega_balanced_target_shape(aspect_ratio, image_resolution, patch_size)
    elif mode == "max_size":
        target_h, target_w = _omega_max_size_target_shape(aspect_ratio, image_resolution, patch_size)
    else:
        raise ValueError("VGGT-Omega preprocess mode must be balanced or max_size.")
    image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    return tvf.ToTensor()(image).to(dtype=torch.float32)


class VFMPreprocessCache:
    """
    CPU-only image preprocess cache with model-specific batch assembly rules.
    """

    def __init__(self, *, model_name: str, max_images: int = 1500) -> None:
        self.model_name = _canonical_model_name(model_name)
        self.max_images = max(1, int(max_images))
        self._cache: OrderedDict[tuple, tuple[Any, int]] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_bytes = 0
        self.last_batch: Dict[str, object] = {}
        self.prefetch_submitted = 0
        self.prefetch_completed = 0
        self.prefetch_skipped_cached = 0
        self.prefetch_errors = 0

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.current_bytes = 0

    def stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "enabled": True,
                "device": "cpu",
                "dtype": "float32",
                "model_name": self.model_name,
                "max_images": int(self.max_images),
                "current_images": int(len(self._cache)),
                "hits": int(self.hits),
                "misses": int(self.misses),
                "evictions": int(self.evictions),
                "estimated_bytes": int(self.current_bytes),
                "last_batch": dict(self.last_batch),
                "prefetch": {
                    "submitted": int(self.prefetch_submitted),
                    "completed": int(self.prefetch_completed),
                    "skipped_cached": int(self.prefetch_skipped_cached),
                    "errors": int(self.prefetch_errors),
                },
            }

    def _evict_locked(self) -> None:
        while len(self._cache) > self.max_images:
            _, (_, byte_count) = self._cache.popitem(last=False)
            self.current_bytes -= int(byte_count)
            self.evictions += 1

    def _get_or_create(self, key: tuple, factory) -> tuple[Any, bool]:
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self.hits += 1
                return _clone_cached(cached[0]), True
            self.misses += 1

        value = factory()
        byte_count = _tensor_bytes(value)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return _clone_cached(cached[0]), True
            self._cache[key] = (_clone_cached(value), byte_count)
            self.current_bytes += int(byte_count)
            self._evict_locked()
        return value, False

    def _finish_batch(self, *, model_name: str, hits_before: int, misses_before: int, evictions_before: int, **extra) -> None:
        with self._lock:
            self.last_batch = {
                "model_name": model_name,
                "hits": int(self.hits - hits_before),
                "misses": int(self.misses - misses_before),
                "evictions": int(self.evictions - evictions_before),
                "current_images": int(len(self._cache)),
                "estimated_bytes": int(self.current_bytes),
                "device": "cpu",
                **extra,
            }

    def load_batch(self, model_name: str, image_paths: Sequence[str], **kwargs):
        model_name = _canonical_model_name(model_name)
        image_paths = [str(path) for path in image_paths]
        with self._lock:
            hits_before = self.hits
            misses_before = self.misses
            evictions_before = self.evictions
        if not image_paths:
            self._finish_batch(
                model_name=model_name,
                hits_before=hits_before,
                misses_before=misses_before,
                evictions_before=evictions_before,
                num_images=0,
            )
            return torch.empty(0, dtype=torch.float32)

        if model_name in {"Pi3", "Pi3x"}:
            batch = self._load_pi3_batch(image_paths, **kwargs)
        elif model_name == "VGGT":
            batch = self._load_vggt_batch(image_paths, **kwargs)
        elif model_name == "VGGTOmega":
            batch = self._load_vggt_omega_batch(image_paths, **kwargs)
        elif model_name == "MapAnything":
            batch = self._load_mapanything_batch(image_paths, **kwargs)
        else:
            raise ValueError(f"Unsupported VFM preprocess cache model: {model_name}")
        self._finish_batch(
            model_name=model_name,
            hits_before=hits_before,
            misses_before=misses_before,
            evictions_before=evictions_before,
            num_images=int(len(image_paths)),
        )
        return batch

    def prefetch_batch(self, image_paths: Sequence[str]) -> Dict[str, int]:
        image_paths = [str(path) for path in image_paths]
        stats = {
            "submitted": int(len(image_paths)),
            "completed": 0,
            "skipped_cached": 0,
            "errors": 0,
        }
        with self._lock:
            self.prefetch_submitted += int(len(image_paths))
            hits_before = self.hits
            misses_before = self.misses
        try:
            self.load_batch(self.model_name, image_paths)
        except Exception:
            with self._lock:
                self.prefetch_errors += int(len(image_paths))
            stats["errors"] = int(len(image_paths))
            return stats
        with self._lock:
            hits_delta = int(self.hits - hits_before)
            misses_delta = int(self.misses - misses_before)
            self.prefetch_skipped_cached += hits_delta
            self.prefetch_completed += misses_delta
        stats["skipped_cached"] = hits_delta
        stats["completed"] = misses_delta
        return stats

    def _load_pi3_batch(self, image_paths: Sequence[str], pixel_limit: int = 255000) -> torch.Tensor:
        with Image.open(image_paths[0]) as first_img:
            target_w, target_h = _pi3_target_size_from_image(first_img, pixel_limit)
        tensors = []
        for image_path in image_paths:
            key = ("Pi3", image_path, int(target_w), int(target_h))
            tensor, _ = self._get_or_create(
                key,
                lambda image_path=image_path: tvf.ToTensor()(
                    _load_rgb_image(image_path).resize((target_w, target_h), Image.Resampling.LANCZOS)
                ).to(dtype=torch.float32),
            )
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    def _load_vggt_batch(self, image_paths: Sequence[str], mode: str = "crop") -> torch.Tensor:
        tensors = []
        for image_path in image_paths:
            key = ("VGGT", image_path, mode)
            tensor, _ = self._get_or_create(
                key,
                lambda image_path=image_path: _vggt_preprocess_one(image_path, mode),
            )
            tensors.append(tensor)
        return torch.stack(_pad_tensors_to_common_size(tensors), dim=0)

    def _load_vggt_omega_batch(
        self,
        image_paths: Sequence[str],
        mode: str = "balanced",
        image_resolution: int = 512,
        patch_size: int = 16,
    ) -> torch.Tensor:
        tensors = []
        for image_path in image_paths:
            key = ("VGGTOmega", image_path, mode, int(image_resolution), int(patch_size))
            tensor, _ = self._get_or_create(
                key,
                lambda image_path=image_path: _omega_preprocess_one(
                    image_path,
                    mode=mode,
                    image_resolution=image_resolution,
                    patch_size=patch_size,
                ),
            )
            tensors.append(tensor)
        return torch.stack(_pad_tensors_to_common_size(tensors), dim=0)

    def _load_mapanything_batch(
        self,
        image_paths: Sequence[str],
        resize_mode: str = "fixed_mapping",
        norm_type: str = "dinov2",
        resolution_set: int = 518,
    ) -> list[dict]:
        from mapanything.utils.cropping import crop_resize_if_necessary
        from mapanything.utils.image import find_closest_aspect_ratio
        from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

        loaded_sizes = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                width, height = img.size
            loaded_sizes.append((width, height))
        average_aspect_ratio = float(sum(width / max(height, 1) for width, height in loaded_sizes) / len(loaded_sizes))
        if resize_mode != "fixed_mapping":
            raise ValueError("MapAnything cached preprocessing currently supports fixed_mapping only.")
        target_width, target_height = find_closest_aspect_ratio(average_aspect_ratio, resolution_set)
        if norm_type not in IMAGE_NORMALIZATION_DICT:
            raise ValueError(f"Unknown MapAnything normalization type: {norm_type}")
        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        transform = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)])

        views = []
        for idx, image_path in enumerate(image_paths):
            key = ("MapAnything", image_path, int(target_width), int(target_height), norm_type)

            def factory(image_path=image_path):
                image = _load_rgb_image(image_path)
                image = crop_resize_if_necessary(image, resolution=(target_width, target_height))[0]
                return {
                    "img": transform(image)[None].to(dtype=torch.float32),
                    "true_shape": np.int32([image.size[::-1]]),
                    "data_norm_type": [norm_type],
                }

            view, _ = self._get_or_create(key, factory)
            view["idx"] = int(idx)
            view["instance"] = str(idx)
            views.append(view)
        return views


class PreprocessingMixin:
    def _merge_image_preprocess_prefetch_stats(self, stats: Dict[str, int]) -> None:
        for key in ("completed", "skipped_cached", "errors"):
            self._image_preprocess_prefetch_stats[key] = int(
                self._image_preprocess_prefetch_stats.get(key, 0)
            ) + int(stats.get(key, 0))

    def _poll_image_preprocess_prefetch(self) -> None:
        if not self._image_preprocess_prefetch_futures:
            self._image_preprocess_prefetch_stats["current_pending"] = 0
            return
        pending: List[Future[Dict[str, int]]] = []
        for future in self._image_preprocess_prefetch_futures:
            if not future.done():
                pending.append(future)
                continue
            try:
                self._merge_image_preprocess_prefetch_stats(dict(future.result()))
            except Exception:
                self._image_preprocess_prefetch_stats["errors"] = int(
                    self._image_preprocess_prefetch_stats.get("errors", 0)
                ) + 1
        self._image_preprocess_prefetch_futures = pending
        self._image_preprocess_prefetch_stats["current_pending"] = int(len(pending))

    def prefetch_image_preprocess(self, image_paths: Sequence[str]) -> None:
        paths = list(image_paths)[: max(0, int(self.config.image_preprocess_prefetch_count))]
        if not paths:
            return
        self._poll_image_preprocess_prefetch()
        if self._image_preprocess_prefetch_executor is None:
            return
        max_pending = max(1, int(self.config.image_preprocess_prefetch_workers))
        if len(self._image_preprocess_prefetch_futures) >= max_pending:
            self._image_preprocess_prefetch_stats["skipped_pending"] = int(
                self._image_preprocess_prefetch_stats.get("skipped_pending", 0)
            ) + len(paths)
            self._image_preprocess_prefetch_stats["current_pending"] = int(
                len(self._image_preprocess_prefetch_futures)
            )
            return
        self._image_preprocess_prefetch_stats["submitted"] = int(
            self._image_preprocess_prefetch_stats.get("submitted", 0)
        ) + len(paths)
        future = self._image_preprocess_prefetch_executor.submit(
            self.runner.prefetch_image_preprocess,
            paths,
        )
        self._image_preprocess_prefetch_futures.append(future)
        self._image_preprocess_prefetch_stats["current_pending"] = int(
            len(self._image_preprocess_prefetch_futures)
        )

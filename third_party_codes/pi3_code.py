"""Minimal Pi3/MoGe helpers required by the on_the_fly3r runtime.

The image loader is adapted from Pi3:
https://github.com/yyfz/Pi3/blob/main/pi3/utils/basic.py

The focal recovery routines are adapted from MoGe:
https://github.com/microsoft/MoGe/blob/main/moge/utils/geometry_torch.py
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def _pi3_target_size_from_image(first_img, pixel_limit):
    width_orig, height_orig = first_img.size
    scale = math.sqrt(pixel_limit / (width_orig * height_orig)) if width_orig * height_orig > 0 else 1
    width_target, height_target = width_orig * scale, height_orig * scale
    width_patches, height_patches = round(width_target / 14), round(height_target / 14)
    while (width_patches * 14) * (height_patches * 14) > pixel_limit:
        if width_patches / height_patches > width_target / height_target:
            width_patches -= 1
        else:
            height_patches -= 1
    return max(1, width_patches) * 14, max(1, height_patches) * 14


def load_images_as_tensor(image_paths, PIXEL_LIMIT=255000, cache=None):
    """Load, uniformly resize, and stack images as an ``[N, 3, H, W]`` tensor."""
    if cache is not None:
        return cache.load_batch(image_paths, pixel_limit=PIXEL_LIMIT)

    sources = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    first_img = sources[0]
    target_width, target_height = _pi3_target_size_from_image(first_img, PIXEL_LIMIT)

    tensor_list = []
    to_tensor = transforms.ToTensor()
    for image in sources:
        try:
            resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            tensor_list.append(to_tensor(resized))
        except Exception as exc:
            print(f"Error processing an image: {exc}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)
    return torch.stack(tensor_list, dim=0)


def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return normalized view-plane coordinates for every image pixel."""
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5
    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def solve_focal_no_shift(uv: np.ndarray, xyz: np.ndarray, z_eps: float = 1e-6) -> float:
    """Solve ``min ||f (xy/z) - uv||_2`` in closed form with zero shift."""
    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    valid = np.isfinite(z) & (np.abs(z) > z_eps)
    if valid.sum() < 2:
        return 1.0

    a = xy[valid] / z[valid, None]
    b = uv[valid]
    numerator = (a * b).sum()
    denominator = (a * a).sum()
    if denominator <= 0 or not np.isfinite(numerator / denominator):
        return 1.0
    return float(numerator / denominator)


@torch.no_grad()
def recover_focal_no_shift(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    downsample_size: Tuple[int, int] = (64, 64),
):
    """Estimate focal length from camera-coordinate points while assuming zero shift."""
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    point_batches = points.reshape(-1, *shape[-3:])
    mask_batches = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    points_low_res = F.interpolate(
        point_batches.permute(0, 3, 1, 2), downsample_size, mode="nearest"
    ).permute(0, 2, 3, 1)

    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)
    uv_low_res = F.interpolate(
        uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest"
    ).squeeze(0).permute(1, 2, 0)
    mask_low_res = None
    if mask_batches is not None:
        mask_low_res = (
            F.interpolate(mask_batches.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest").squeeze(1)
            > 0
        )

    uv_numpy = uv_low_res.cpu().numpy()
    points_numpy = points_low_res.detach().cpu().numpy()
    mask_numpy = None if mask_low_res is None else mask_low_res.cpu().numpy()

    focals = []
    for index in range(points_numpy.shape[0]):
        if mask_numpy is None:
            uv_values = uv_numpy.reshape(-1, 2)
            xyz_values = points_numpy[index].reshape(-1, 3)
        else:
            valid = mask_numpy[index].reshape(-1)
            uv_values = uv_numpy.reshape(-1, 2)[valid]
            xyz_values = points_numpy[index].reshape(-1, 3)[valid]

        focal = 1.0 if uv_values.shape[0] < 2 else solve_focal_no_shift(uv_values, xyz_values)
        focals.append(focal)

    return torch.tensor(focals, device=points.device, dtype=points.dtype).reshape(shape[:-3])

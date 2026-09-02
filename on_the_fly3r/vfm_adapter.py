import os
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from third_party_codes.vggt_code import apply_sky_segmentation

from .config import ReconstructionConfig
from .types import SubsetInferenceResult
from .utils import _resolve_runtime_device
from .preprocessing import VFMPreprocessCache


DEFAULT_PI3_CHECKPOINT_ENV = "ON_THE_FLY3R_PI3_CHECKPOINT"
DEFAULT_PI3_CHECKPOINT_PATH = "/home/shenzhe/Pi3/checkpoints/model.safetensors"
DEFAULT_PI3X_CHECKPOINT_PATH = str(
    Path(__file__).resolve().parents[1] / "Pi3/Pi3XCheckpoints/model.safetensors"
)
DEFAULT_VGGT_CHECKPOINT = "/data2/sz/VGGT_checkpoints/model.pt"
DEFAULT_MAPANYTHING_CHECKPOINT = "/data2/sz/checkpoints/MapAnything"
DEFAULT_VGGT_OMEGA_CHECKPOINT = "/data2/sz/VGGT_omega_checkpoints/vggt_omega_1b_512.pt"


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


def _resolve_pi3_checkpoint(model_name: str, model_checkpoint):
    checkpoint = model_checkpoint
    if model_name == "Pi3x" and (not checkpoint or str(checkpoint) == DEFAULT_PI3_CHECKPOINT_PATH):
        checkpoint = DEFAULT_PI3X_CHECKPOINT_PATH
    if not checkpoint:
        checkpoint = os.environ.get(DEFAULT_PI3_CHECKPOINT_ENV)
    if checkpoint:
        return checkpoint
    raise ValueError(
        f"Model '{model_name}' requires --model_checkpoint or the "
        f"{DEFAULT_PI3_CHECKPOINT_ENV} environment variable."
    )


def _resolve_default_local_checkpoint(model_name: str, model_checkpoint, default_checkpoint: str):
    checkpoint = model_checkpoint
    if not checkpoint or str(checkpoint) == DEFAULT_PI3_CHECKPOINT_PATH:
        checkpoint = default_checkpoint
    if checkpoint:
        return checkpoint
    raise ValueError(f"{model_name} requires a local checkpoint path.")


def _resolve_vggt_checkpoint(model_checkpoint):
    return _resolve_default_local_checkpoint("VGGT", model_checkpoint, DEFAULT_VGGT_CHECKPOINT)


def _resolve_mapanything_checkpoint(model_checkpoint):
    return _resolve_default_local_checkpoint("MapAnything", model_checkpoint, DEFAULT_MAPANYTHING_CHECKPOINT)


def _resolve_vggt_omega_checkpoint(model_checkpoint):
    return _resolve_default_local_checkpoint("VGGTOmega", model_checkpoint, DEFAULT_VGGT_OMEGA_CHECKPOINT)


def _load_state_dict_from_checkpoint(checkpoint, model_name: str):
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"{model_name} checkpoint not found: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            nested = state_dict.get(key)
            if isinstance(nested, dict):
                return nested
    return state_dict


def _ensure_local_vggt_omega_import_path() -> None:
    local_root = Path(__file__).resolve().parent / "vggt-omega"
    if local_root.exists():
        path = str(local_root)
        if path not in sys.path:
            sys.path.insert(0, path)


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _register_feature_from_tokens(tokens, *, source: str):
    if tokens is None or not torch.is_tensor(tokens):
        return None, {}
    register_tokens = tokens.detach().float()
    if register_tokens.ndim == 4:
        # [B, S, R, D] -> [S, D]
        if register_tokens.shape[0] <= 0 or register_tokens.shape[2] <= 0:
            return None, {}
        features = register_tokens.mean(dim=2)[0].cpu().numpy().astype(np.float32, copy=False)
        metadata = {
            "source": source,
            "num_register_tokens": int(register_tokens.shape[2]),
            "feature_dim": int(register_tokens.shape[-1]),
        }
        return features, metadata
    if register_tokens.ndim == 3:
        # [S, R, D] -> [S, D]
        if register_tokens.shape[1] <= 0:
            return None, {}
        features = register_tokens.mean(dim=1).cpu().numpy().astype(np.float32, copy=False)
        metadata = {
            "source": source,
            "num_register_tokens": int(register_tokens.shape[1]),
            "feature_dim": int(register_tokens.shape[-1]),
        }
        return features, metadata
    return None, {}


def _load_with_cache_or_loader(image_preprocess_cache, model_name, image_names, loader, **cache_kwargs):
    if image_preprocess_cache is not None:
        return image_preprocess_cache.load_batch(model_name, image_names, **cache_kwargs)
    return loader(image_names)

def initialize_model(args, device):
    model_name = _canonical_model_name(getattr(args, "model", "Pi3"))
    model_checkpoint = getattr(args, "model_checkpoint", None)

    if model_name == "VGGT":
        from vggt.models.vggt import VGGT

        checkpoint = _resolve_vggt_checkpoint(model_checkpoint)
        model = VGGT()
        model.load_state_dict(_load_state_dict_from_checkpoint(checkpoint, model_name))
        model = model.to(device).eval()

    elif model_name == "Pi3":
        from Pi3.pi3.models.pi3 import Pi3 as Pi3Model
        checkpoint = _resolve_pi3_checkpoint(model_name, model_checkpoint)
        if str(checkpoint).endswith(".safetensors"):
            from safetensors.torch import load_file

            model = Pi3Model().to(device).eval()
            model.load_state_dict(load_file(checkpoint, device=device))
        else:
            model = Pi3Model.from_pretrained(checkpoint).to(device).eval()
    elif model_name == "Pi3x":
        from Pi3.pi3.models.pi3x import Pi3X

        model_checkpoint = _resolve_pi3_checkpoint(model_name, model_checkpoint)
        if str(model_checkpoint).endswith(".safetensors"):
            from safetensors.torch import load_file

            model = Pi3X().to(device).eval()
            model.load_state_dict(load_file(model_checkpoint, device=device), strict=False)
        else:
            model = Pi3X.from_pretrained(model_checkpoint).to(device).eval()
    elif model_name == "MapAnything":
        from mapanything.models import MapAnything
        checkpoint = _resolve_mapanything_checkpoint(model_checkpoint)
        model = MapAnything.from_pretrained(checkpoint).to(device).eval()
    elif model_name == "VGGTOmega":
        _ensure_local_vggt_omega_import_path()
        from vggt_omega.models import VGGTOmega

        checkpoint = _resolve_vggt_omega_checkpoint(model_checkpoint)
        model = VGGTOmega().to(device).eval()
        state_dict = _load_state_dict_from_checkpoint(checkpoint, model_name)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")
    return model


def _sync_cuda_for_profile(device) -> None:
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError):
        return
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)


def run_predictions(
    image_names,
    model_name,
    model,
    sky_mask=False,
    return_register_tokens=False,
    return_profile=False,
    image_preprocess_cache=None,
):
    model_name = _canonical_model_name(model_name)
    if model_name == "VGGT":
        results = run_VGGT(
            image_names,
            model,
            sky_mask,
            return_register_tokens=return_register_tokens,
            image_preprocess_cache=image_preprocess_cache,
        )
    elif model_name in {"Pi3", "Pi3x"}:
        results = run_Pi3(
            image_names,
            model,
            sky_mask,
            return_register_tokens=return_register_tokens,
            return_profile=return_profile,
            image_preprocess_cache=image_preprocess_cache,
        )
    elif model_name == "MapAnything":
        results = run_MapAnything(
            image_names,
            model,
            sky_mask,
            return_register_tokens=return_register_tokens,
            image_preprocess_cache=image_preprocess_cache,
        )
    elif model_name == "VGGTOmega":
        results = run_VGGTOmega(
            image_names,
            model,
            sky_mask,
            return_register_tokens=return_register_tokens,
            return_profile=return_profile,
            image_preprocess_cache=image_preprocess_cache,
        )
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'.")
    return results


class VFMInferenceRunner:
    """
    Lightweight wrapper around VFM inference with a model resident on one device.
    """

    def __init__(self, config: ReconstructionConfig, model_args) -> None:
        self.config = config
        self.model_args = model_args
        self.device = _resolve_runtime_device(config.inference_device)
        self.model = load_vfm(model_args, self.device)
        self.image_preprocess_cache = VFMPreprocessCache(
            model_name=self.config.model_name,
            max_images=self.config.image_preprocess_cache_max_images,
        )

    def infer_subset(self, image_paths: Sequence[str]) -> SubsetInferenceResult:
        frame_ids = [Path(path).stem for path in image_paths]
        should_capture_register = bool(self.config.enable_register_ref_pruning_retry)
        outputs = infer_vfm(
            list(image_paths),
            self.config.model_name,
            self.model,
            sky_mask=self.config.use_sky_mask,
            return_register_tokens=should_capture_register,
            return_profile=self.config.enable_detailed_profiling,
            image_preprocess_cache=self.image_preprocess_cache,
        )
        vfm_register_tokens = outputs.get("vfm_register_tokens")
        if vfm_register_tokens is not None:
            vfm_register_tokens = np.asarray(vfm_register_tokens, dtype=np.float32)

        return SubsetInferenceResult(
            image_paths=list(image_paths),
            frame_ids=frame_ids,
            cam2world=np.asarray(outputs["cam2world"], dtype=np.float32),
            intrinsic=np.asarray(outputs["intrinsic"], dtype=np.float32),
            world_points=np.asarray(outputs["world_points"], dtype=self.config.point_dtype),
            world_points_conf=np.asarray(outputs["world_points_conf"], dtype=self.config.conf_dtype),
            images=np.asarray(outputs["images"], dtype=self.config.image_dtype),
            vfm_register_tokens=vfm_register_tokens,
            vfm_register_metadata=dict(outputs.get("vfm_register_metadata", {})),
            profile=dict(outputs.get("profile", {})),
        )

    def image_preprocess_cache_stats(self) -> Dict[str, object]:
        return dict(self.image_preprocess_cache.stats())

    def prefetch_image_preprocess(self, image_paths: Sequence[str]) -> Dict[str, int]:
        return dict(self.image_preprocess_cache.prefetch_batch(list(image_paths)))

    def clear_image_preprocess_cache(self) -> None:
        self.image_preprocess_cache.clear()

    def release_gpu_resources(self) -> None:
        """Drop the resident reconstruction model after inference is complete."""
        self.model = None



def load_vfm(model_args, device):
    """Load one supported VFM on the requested runtime device."""
    return initialize_model(model_args, device)


def infer_vfm(image_names, model_name, model, **kwargs):
    """Run a supported VFM and return the established reconstruction fields."""
    return run_predictions(image_names, model_name, model, **kwargs)

def run_VGGT(
    image_names,
    model,
    sky_mask=False,
    return_register_tokens=False,
    image_preprocess_cache=None,
):
    results = {}
    from vggt.utils.load_fn import load_and_preprocess_images as load_images
    from vggt.utils.geometry import closed_form_inverse_se3
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = _load_with_cache_or_loader(image_preprocess_cache, "VGGT", image_names, load_images)
    model_device = _model_device(model)
    images = images.to(model_device)
    results['org_images'] = images
    model_input = images[None]
    with torch.no_grad():
        if model_device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability(model_device.index or 0)[0] >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, patch_start_idx = model.aggregator(model_input)
        else:
            aggregated_tokens_list, patch_start_idx = model.aggregator(model_input)
        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            if model.camera_head is not None:
                pose_enc_list = model.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
            if model.point_head is not None:
                pts3d, pts3d_conf = model.point_head(
                    aggregated_tokens_list,
                    images=model_input,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

    if return_register_tokens and patch_start_idx > 1:
        final_tokens = aggregated_tokens_list[-1]
        features, metadata = _register_feature_from_tokens(
            final_tokens[:, :, 1:patch_start_idx],
            source="vggt_aggregator_register_mean",
        )
        results["vfm_register_tokens"] = features
        results["vfm_register_metadata"] = metadata
    results["images"] = (images.permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    results["cam2world"] = closed_form_inverse_se3(extrinsic.cpu().numpy().squeeze(0))
    results["intrinsic"] = intrinsic.cpu().numpy().squeeze(0)
    results['world_points'] = predictions["world_points"].cpu().numpy().squeeze(0)  # (S, H, W, 3)
    results["world_points_conf"] = predictions["world_points_conf"].cpu().numpy().squeeze(0)
    if sky_mask:
        non_sky_mask_binary = apply_sky_segmentation(image_names, images.shape[-2:])  # (n, H, W)
        results["world_points_conf"] = (non_sky_mask_binary * results["world_points_conf"])

    return results



def _install_pi3_register_capture_hooks(model, num_frames):
    captures = {"last_odd_register_mean": None}
    handles = []
    register_count = int(getattr(model, "patch_start_idx", 0))
    decoder = getattr(model, "decoder", None)

    if register_count <= 0 or decoder is None:
        return handles, captures

    decoder_blocks = list(decoder)
    odd_indices = [idx for idx in range(len(decoder_blocks)) if idx % 2 == 1]
    if not odd_indices:
        return handles, captures
    target_block_idx = odd_indices[-1]

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return
        if num_frames <= 0 or hidden.shape[1] % num_frames != 0:
            return
        token_length = hidden.shape[1] // num_frames
        if token_length < register_count:
            return
        batch_size = hidden.shape[0]
        register = hidden.reshape(batch_size, num_frames, token_length, -1)[:, :, :register_count]
        captures["last_odd_register_mean"] = register.detach().float().mean(dim=2).cpu()

    handles.append(decoder_blocks[target_block_idx].register_forward_hook(hook))
    return handles, captures


def run_Pi3(
    image_names,
    model,
    sky_mask=False,
    return_register_tokens=False,
    return_profile=False,
    image_preprocess_cache=None,
):
    results = {}
    profile = {}
    from Pi3.pi3.utils.geometry import homogenize_points
    from third_party_codes.pi3_code import load_images_as_tensor as load_images, recover_focal_no_shift
    total_start = time.perf_counter()
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        try:
            model_device = next(model.buffers()).device
        except StopIteration:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_start = time.perf_counter()
    if image_preprocess_cache is not None:
        images = image_preprocess_cache.load_batch("Pi3", image_names, pixel_limit=255000)
    else:
        images = load_images(image_names)
    if return_profile:
        profile["load_images_cpu_sec"] = time.perf_counter() - stage_start
        if image_preprocess_cache is not None:
            cache_batch = dict(getattr(image_preprocess_cache, "last_batch", {}) or {})
            profile["image_cache_hits"] = float(cache_batch.get("hits", 0))
            profile["image_cache_misses"] = float(cache_batch.get("misses", 0))
            profile["image_cache_evictions"] = float(cache_batch.get("evictions", 0))
            profile["image_cache_current_images"] = float(cache_batch.get("current_images", 0))
            profile["image_cache_estimated_bytes"] = float(cache_batch.get("estimated_bytes", 0))
            profile["image_cache_target_height"] = float(cache_batch.get("target_height", 0))
            profile["image_cache_target_width"] = float(cache_batch.get("target_width", 0))

    stage_start = time.perf_counter()
    images = images.to(model_device)
    if return_profile:
        _sync_cuda_for_profile(model_device)
        profile["to_device_sec"] = time.perf_counter() - stage_start
    results['org_images'] = images
    images = images[None]

    hook_handles = []
    register_captures = {"last_odd_register_mean": None}
    if return_register_tokens:
        hook_handles, register_captures = _install_pi3_register_capture_hooks(model, len(image_names))

    try:
        with torch.no_grad():
            if return_profile:
                _sync_cuda_for_profile(model_device)
            stage_start = time.perf_counter()
            if model_device.type == "cuda":
                dtype = torch.bfloat16 if torch.cuda.get_device_capability(model_device.index or 0)[0] >= 8 else torch.float16
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)
            else:
                predictions = model(images)
            if return_profile:
                _sync_cuda_for_profile(model_device)
                profile["model_forward_sec"] = time.perf_counter() - stage_start
    finally:
        for handle in hook_handles:
            handle.remove()

    if return_register_tokens:
        register_mean = register_captures.get("last_odd_register_mean")
        if register_mean is not None and register_mean.ndim == 3 and register_mean.shape[0] > 0:
            features = register_mean[0].numpy().astype(np.float32, copy=False)
            results["vfm_register_tokens"] = features
            results["vfm_register_metadata"] = {
                "source": "pi3_decoder_register_mean",
                "num_register_tokens": int(getattr(model, "patch_start_idx", 0)),
                "feature_dim": int(features.shape[-1]) if features.ndim == 2 else 0,
            }
        else:
            results["vfm_register_tokens"] = None
            results["vfm_register_metadata"] = {"source": "pi3_decoder_register_mean", "missing_reason": "hook_capture_empty"}

    if return_profile:
        _sync_cuda_for_profile(model_device)
    stage_start = time.perf_counter()
    cam2world = torch.inverse(predictions['camera_poses'][0][0]) @ predictions['camera_poses'][0] # (S, 4, 4)
    if return_profile:
        _sync_cuda_for_profile(model_device)
        profile["cam2world_gpu_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    results['cam2world'] = cam2world.cpu().numpy() # (S, 4, 4)
    if return_profile:
        profile["cam2world_cpu_copy_sec"] = time.perf_counter() - stage_start

    if return_profile:
        _sync_cuda_for_profile(model_device)
    stage_start = time.perf_counter()
    world_points = torch.einsum('nij, nhwj -> nhwi', cam2world, homogenize_points(predictions['local_points'][0]))[..., :3]
    if return_profile:
        _sync_cuda_for_profile(model_device)
        profile["world_points_gpu_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    results['world_points'] = world_points.cpu().numpy() # (S, H, W, 3)
    if return_profile:
        profile["world_points_cpu_copy_sec"] = time.perf_counter() - stage_start

    if return_profile:
        _sync_cuda_for_profile(model_device)
    stage_start = time.perf_counter()
    world_points_conf = torch.sigmoid(predictions['conf'][0,...,0])
    if return_profile:
        _sync_cuda_for_profile(model_device)
        profile["confidence_gpu_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    results['world_points_conf'] = world_points_conf.cpu().numpy() # (S, H, W)
    if return_profile:
        profile["confidence_cpu_copy_sec"] = time.perf_counter() - stage_start

    if sky_mask:
        stage_start = time.perf_counter()
        non_sky_mask_binary = apply_sky_segmentation(image_names, images.shape[-2:])  # (n, H, W)
        results["world_points_conf"] = non_sky_mask_binary * results["world_points_conf"]
        if return_profile:
            profile["sky_mask_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    results["images"] = (images[0].permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)
    if return_profile:
        profile["images_cpu_copy_sec"] = time.perf_counter() - stage_start


    points = predictions["local_points"] # (1, S, H, W, 3)
    masks = None
    original_height, original_width = points.shape[-3:-1]
    aspect_ratio = original_width / original_height
    # use recover_focal_shift function from MoGe
    stage_start = time.perf_counter()
    focal = recover_focal_no_shift(points, masks) # focal: (1, S), shift: (1, S)
    if return_profile:
        _sync_cuda_for_profile(model_device)
        profile["recover_focal_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
    zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
    zero_point_five = torch.full_like(fx, 0.5)
    intrinsics = torch.stack([
        fx, zeros, zero_point_five,
        zeros, fy, zero_point_five,
        zeros, zeros, ones
    ], dim=-1).reshape(-1, 3, 3) # (S, 3, 3)


    H, W = original_height, original_width

    intrinsics[:, 0, 0] *= (W - 1)   # fx
    intrinsics[:, 1, 1] *= (H - 1)   # fy
    intrinsics[:, 0, 2] *= (W - 1)   # cx
    intrinsics[:, 1, 2] *= (H - 1)   # cy
    if return_profile:
        _sync_cuda_for_profile(model_device)
        profile["intrinsic_gpu_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    results['intrinsic'] = intrinsics.cpu().numpy()
    if return_profile:
        profile["intrinsic_cpu_copy_sec"] = time.perf_counter() - stage_start
        profile["num_images"] = float(len(image_names))
        profile["input_height"] = float(original_height)
        profile["input_width"] = float(original_width)
        profile["total_sec"] = time.perf_counter() - total_start
        results["profile"] = profile

    return results




def _mapanything_register_capture(model, enabled: bool):
    if not enabled or not hasattr(model, "info_sharing"):
        return None, {}
    captures = {"final_info_sharing_features": None}
    original = model.info_sharing.forward

    def wrapped(*args, **kwargs):
        output = original(*args, **kwargs)
        final_output = output[0] if isinstance(output, tuple) else output
        captures["final_info_sharing_features"] = getattr(final_output, "features", None)
        return output

    model.info_sharing.forward = wrapped
    return original, captures


def _restore_mapanything_register_capture(model, original) -> None:
    if original is not None and hasattr(model, "info_sharing"):
        model.info_sharing.forward = original


def _mapanything_register_features(captures):
    if not captures:
        return None, {}
    final_features = captures.get("final_info_sharing_features")
    if final_features is None:
        return None, {
            "source": "mapanything_final_info_sharing_gap",
            "missing_reason": "final_info_sharing_features_not_captured",
        }
    per_view = []
    feature_dim = None
    for feature in final_features:
        if feature is None or not torch.is_tensor(feature):
            return None, {
                "source": "mapanything_final_info_sharing_gap",
                "missing_reason": "invalid_final_info_sharing_tensor",
            }
        tensor = feature.detach().float()
        if tensor.ndim == 4:
            if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
                return None, {
                    "source": "mapanything_final_info_sharing_gap",
                    "missing_reason": "empty_final_info_sharing_tensor",
                }
            pooled = tensor.mean(dim=(2, 3))
            feature_dim = int(pooled.shape[-1])
            per_view.append(pooled.mean(dim=0))
        elif tensor.ndim == 3:
            if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
                return None, {
                    "source": "mapanything_final_info_sharing_gap",
                    "missing_reason": "empty_final_info_sharing_tensor",
                }
            pooled = tensor.mean(dim=2)
            feature_dim = int(pooled.shape[-1])
            per_view.append(pooled.mean(dim=0))
        elif tensor.ndim == 2:
            feature_dim = int(tensor.shape[-1])
            per_view.append(tensor.mean(dim=0))
        else:
            return None, {
                "source": "mapanything_final_info_sharing_gap",
                "missing_reason": "unsupported_final_info_sharing_shape",
                "feature_shape": list(tensor.shape),
            }
    if not per_view:
        return None, {
            "source": "mapanything_final_info_sharing_gap",
            "missing_reason": "empty_final_info_sharing_list",
        }
    features = torch.stack(per_view, dim=0).cpu().numpy().astype(np.float32, copy=False)
    return features, {
        "source": "mapanything_final_info_sharing_gap",
        "uses_register_tokens": False,
        "feature_kind": "final_info_sharing_feature",
        "feature_dim": int(feature_dim or 0),
    }


def _unproject_depth_to_world_points(depth_map: np.ndarray, cam2world: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )
    return (
        np.einsum("sij,shwj->shwi", cam2world[:, :3, :3], camera_points)
        + cam2world[:, None, None, :3, 3]
    ).astype(np.float32, copy=False)


def run_VGGTOmega(
    image_names,
    model,
    sky_mask=False,
    return_register_tokens=False,
    return_profile=False,
    image_preprocess_cache=None,
):
    _ensure_local_vggt_omega_import_path()
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    results = {}
    profile = {}
    total_start = time.perf_counter()
    loader = lambda paths: load_and_preprocess_images(paths, image_resolution=512)

    stage_start = time.perf_counter()
    images = _load_with_cache_or_loader(
        image_preprocess_cache,
        "VGGTOmega",
        image_names,
        loader,
        image_resolution=512,
    )
    if return_profile:
        profile["load_images_cpu_sec"] = time.perf_counter() - stage_start

    model_device = _model_device(model)
    stage_start = time.perf_counter()
    images = images.to(model_device)
    if return_profile:
        _sync_cuda_for_profile(model_device)
        profile["to_device_sec"] = time.perf_counter() - stage_start
    results["org_images"] = images

    with torch.inference_mode():
        if return_profile:
            _sync_cuda_for_profile(model_device)
        stage_start = time.perf_counter()
        predictions = model(images)
        if return_profile:
            _sync_cuda_for_profile(model_device)
            profile["model_forward_sec"] = time.perf_counter() - stage_start

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    extrinsic_np = extrinsic.detach().float().cpu().numpy()[0]
    intrinsic_np = intrinsic.detach().float().cpu().numpy()[0]
    extrinsic_4x4 = np.tile(np.eye(4, dtype=np.float32)[None], (extrinsic_np.shape[0], 1, 1))
    extrinsic_4x4[:, :3, :4] = extrinsic_np
    cam2world_abs = np.linalg.inv(extrinsic_4x4)
    cam2world = np.linalg.inv(cam2world_abs[0]) @ cam2world_abs

    depth_np = predictions["depth"].detach().float().cpu().numpy()[0]
    conf_np = predictions["depth_conf"].detach().float().cpu().numpy()[0]
    if conf_np.ndim == 4 and conf_np.shape[-1] == 1:
        conf_np = conf_np[..., 0]
    world_points = _unproject_depth_to_world_points(depth_np, cam2world, intrinsic_np)

    results["cam2world"] = cam2world.astype(np.float32, copy=False)
    results["intrinsic"] = intrinsic_np.astype(np.float32, copy=False)
    results["world_points"] = world_points
    results["world_points_conf"] = conf_np.astype(np.float32, copy=False)
    results["images"] = (
        predictions["images"][0].permute(0, 2, 3, 1).detach().float().cpu().numpy() * 255
    ).clip(0, 255).astype(np.uint8)

    if return_register_tokens:
        camera_and_register_tokens = predictions.get("camera_and_register_tokens")
        register_tokens = None
        if torch.is_tensor(camera_and_register_tokens) and camera_and_register_tokens.shape[2] > 1:
            register_tokens = camera_and_register_tokens[:, :, 1:]
        features, metadata = _register_feature_from_tokens(
            register_tokens,
            source="vggt_omega_camera_and_register_tokens",
        )
        if features is None:
            metadata = {
                "source": "vggt_omega_camera_and_register_tokens",
                "missing_reason": "camera_and_register_tokens_missing_or_empty",
            }
        results["vfm_register_tokens"] = features
        results["vfm_register_metadata"] = metadata

    if sky_mask:
        non_sky_mask_binary = apply_sky_segmentation(image_names, images.shape[-2:])
        results["world_points_conf"] = non_sky_mask_binary * results["world_points_conf"]

    if return_profile:
        profile["num_images"] = float(len(image_names))
        profile["input_height"] = float(images.shape[-2])
        profile["input_width"] = float(images.shape[-1])
        profile["total_sec"] = time.perf_counter() - total_start
        results["profile"] = profile

    return results


def run_MapAnything(
    image_names,
    model,
    sky_mask=False,
    return_register_tokens=False,
    image_preprocess_cache=None,
):
    from mapanything.utils.image import load_images
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = _load_with_cache_or_loader(image_preprocess_cache, "MapAnything", image_names, load_images)
    capture_original, register_captures = _mapanything_register_capture(model, return_register_tokens)
    try:
        predictions = model.infer(
            images,                            # Input views
            memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
            use_amp=True,                     # Use mixed precision inference (recommended)
            amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
            apply_mask=True,                  # Apply masking to dense geometry outputs
            mask_edges=True,                  # Remove edge artifacts by using normals and depth
            apply_confidence_mask=False,      # Filter low-confidence regions
            confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels
        )
    finally:
        _restore_mapanything_register_capture(model, capture_original)
    results = {
        "cam2world": torch.cat([pred["camera_poses"] for pred in predictions], 0).cpu().numpy(),         # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame (B, 4, 4)
        "intrinsic": torch.cat([pred["intrinsics"] for pred in predictions], 0).cpu().numpy(),           # Recovered pinhole camera intrinsics (B, 3, 3)
        "world_points": torch.cat([pred["pts3d"] for pred in predictions], 0).cpu().numpy(),                     # 3D points in world coordinates (B, H, W, 3)
        "world_points_conf": torch.cat([(pred["conf"] * pred["mask"].squeeze(-1)) for pred in predictions], 0).cpu().numpy(),                   # Per-pixel confidence scores (B, H, W)
        "org_images": torch.cat([image['img'] for image in images]).to(device),
        "images": (torch.cat([pred["img_no_norm"] for pred in predictions], 0).cpu().numpy()*255).astype(np.uint8)                  # Denormalized input images for visualization (B, H, W, 3)
    }
    if return_register_tokens:
        features, metadata = _mapanything_register_features(register_captures)
        results["vfm_register_tokens"] = features
        results["vfm_register_metadata"] = metadata
    results['cam2world'] = np.linalg.inv(results['cam2world'][0]) @ results['cam2world'] # (S, 4, 4)
    if sky_mask:
        non_sky_mask_binary = apply_sky_segmentation(image_names, images[0]['true_shape'][0].tolist())  # (n, H, W)
        results["world_points_conf"] = non_sky_mask_binary * results["world_points_conf"]

    return results

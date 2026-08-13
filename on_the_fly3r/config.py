import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


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


@dataclass
class ReconstructionConfig:
    model_name: str = "Pi3"
    init_window: int = 30
    max_batch_size: int = 5
    dynamic_batch_ref_overlap_min: int = 2
    dynamic_batch_ref_overlap_ratio_min: float = 0.4
    dynamic_batch_query_similarity_min: float = 0.75
    dynamic_batch_dominant_refs: int = 5
    retrieval_topk: int = 5
    min_topk_for_alignment: int = 3
    use_sky_mask: bool = False
    inference_device: str = "cuda"
    retrieval_device: str = "cuda"
    supscene_weights: Optional[str] = None
    supscene_image_size: int = 322
    enable_profiling_sync: bool = False
    enable_detailed_profiling: bool = False
    enable_retrieval_prefetch: bool = True
    retrieval_prefetch_count: int = 16
    alignment_use_scale: bool = True
    alignment_point_conf_threshold: float = 0.5
    alignment_grid_topk_per_cell: int = 12
    alignment_grid_conf_min: float = 0.5
    alignment_fine_residual_mad_scale: float = 2.5
    enable_alignment_validation: bool = True
    enable_register_ref_pruning_retry: bool = False
    register_ref_pruning_candidate_z: float = 2.0
    register_ref_pruning_min_similarity: float = 0.0
    register_ref_pruning_max_drop_refs: int = 1
    register_ref_pruning_min_refs: int = 0
    validation_translation_spacing_multiplier: float = 50.0
    validation_spacing_estimation_samples: int = 2048
    validation_spacing_estimation_max_frames: int = 5
    validation_median_rotation_deg_threshold: float = 10.0
    validation_max_rotation_deg_threshold: float = 25.0
    validation_scale_min: float = 0.5
    validation_scale_max: float = 1.5
    fusion_conf_threshold: float = 0.5
    image_preprocess_cache_max_images: int = 1500
    image_preprocess_prefetch_count: int = 20
    image_preprocess_prefetch_workers: int = 1
    alignment_compact_max_points: int = 5000
    alignment_compact_grid_cell_size: Optional[int] = None
    enable_pose_graph_optimization: bool = False
    pose_opt_min_edges: int = 30
    pose_opt_max_nfev: int = 15
    pose_graph_loop_min_separation: int = 50
    pose_graph_continuity_similarity_min: float = 0.65
    pose_graph_continuity_ref_overlap_min: int = 2
    pose_graph_continuity_step_multiplier: float = 4.0
    pose_graph_continuity_rotation_max_deg: float = 45.0
    point_dtype: np.dtype = np.float32
    conf_dtype: np.dtype = np.float32
    image_dtype: np.dtype = np.uint8

    def __post_init__(self) -> None:
        self.model_name = _canonical_model_name(self.model_name)
        self.max_batch_size = max(1, int(self.max_batch_size))
        self.register_ref_pruning_max_drop_refs = max(1, int(self.register_ref_pruning_max_drop_refs))
        self.register_ref_pruning_min_refs = max(0, int(self.register_ref_pruning_min_refs))
        self.alignment_grid_topk_per_cell = max(1, int(self.alignment_grid_topk_per_cell))
        self.alignment_grid_conf_min = max(0.0, float(self.alignment_grid_conf_min))
        self.image_preprocess_cache_max_images = max(1, int(self.image_preprocess_cache_max_images))
        self.image_preprocess_prefetch_count = max(1, int(self.image_preprocess_prefetch_count))
        self.image_preprocess_prefetch_workers = max(1, int(self.image_preprocess_prefetch_workers))
        self.alignment_compact_max_points = max(1, int(self.alignment_compact_max_points))
        if self.alignment_compact_grid_cell_size is None:
            self.alignment_compact_grid_cell_size = 32
        self.alignment_compact_grid_cell_size = max(1, int(self.alignment_compact_grid_cell_size))
        self.pose_opt_min_edges = max(0, int(self.pose_opt_min_edges))
        self.pose_opt_max_nfev = max(1, int(self.pose_opt_max_nfev))
        self.pose_graph_loop_min_separation = max(1, int(self.pose_graph_loop_min_separation))
        self.pose_graph_continuity_ref_overlap_min = max(0, int(self.pose_graph_continuity_ref_overlap_min))
        self.pose_graph_continuity_step_multiplier = max(1.0, float(self.pose_graph_continuity_step_multiplier))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run progressive multi-model reconstruction on an image folder.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config file. CLI flags override config values.")
    parser.add_argument("--case", "--config_case", dest="config_case", type=str, default=None, help="Optional ablation/case name inside the config file.")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing the ordered image sequence.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save logs and fused point clouds. Defaults to output_root/case when configured, otherwise outputs/incremental_run.")
    parser.add_argument("--output_root", type=str, default=None, help="Optional root directory used to derive output_dir as output_root/case.")
    parser.add_argument("--max_images", type=int, default=0, help="Optional cap on the number of ordered images processed. <=0 uses all images.")
    parser.add_argument("--model", type=str, default="Pi3", choices=["VGGT", "Pi3", "Pi3x", "MapAnything", "VGGTOmega", "VGGT-Omega"], help="Forward model used for subset reconstruction.")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="/home/shenzhe/Pi3/checkpoints/model.safetensors",
        help=(
            "Checkpoint path or repo id for the selected model. "
            "Pi3 uses the provided path or ON_THE_FLY3R_PI3_CHECKPOINT; "
            "Pi3x defaults to Pi3/Pi3XCheckpoints/model.safetensors; "
            "VGGT defaults to /data2/sz/VGGT_checkpoints/model.pt; "
            "MapAnything defaults to /data2/sz/checkpoints/MapAnything; "
            "VGGTOmega defaults to /data2/sz/VGGT_omega_checkpoints/vggt_omega_1b_512.pt."
        ),
    )
    parser.add_argument("--init_window", type=int, default=30, help="Number of images used for initialization.")
    parser.add_argument("--max_batch_size", type=int, default=5, help="Maximum number of new images allowed in one dynamically formed batch.")
    parser.add_argument("--dynamic_batch_ref_overlap_min", type=int, default=2, help="Minimum overlap count between a new query's top references and the current batch dominant references.")
    parser.add_argument("--dynamic_batch_ref_overlap_ratio_min", type=float, default=0.4, help="Minimum dominant-reference overlap ratio needed to keep extending a dynamic batch.")
    parser.add_argument("--dynamic_batch_query_similarity_min", type=float, default=0.75, help="Minimum query-query similarity used when extending a dynamic batch.")
    parser.add_argument("--dynamic_batch_dominant_refs", type=int, default=5, help="How many dominant references define the current batch during dynamic batch formation.")
    parser.add_argument("--retrieval_topk", type=int, default=5, help="Number of reconstructed reference images selected for each incoming batch.")
    parser.add_argument("--min_topk_for_alignment", type=int, default=3, help="Minimum shared images required for alignment.")
    parser.add_argument("--use_sky_mask", action="store_true", help="Apply sky masking during Pi3 inference.")
    parser.add_argument("--inference_device", type=str, default="cuda:2", help="Device for forward inference, e.g. cpu, cuda, cuda:0, cuda:1.")
    parser.add_argument("--retrieval_device", type=str, default="cuda:2", help="CUDA device used for SupScene descriptor extraction and Torch retrieval search.")
    parser.add_argument("--supscene_weights", type=str, default=None, help="Released SupScene .pth weight path.")
    parser.add_argument("--supscene_image_size", type=int, default=322, help="Input resolution used by the SupScene retrieval encoder.")
    parser.add_argument("--enable_profiling_sync", action="store_true", help="Enable CUDA synchronization around timed stages for more accurate profiling.")
    parser.add_argument("--enable_detailed_profiling", action="store_true", help="Record fine-grained inference/alignment/fusion timings and save an aggregated profile summary.")
    parser.add_argument("--disable_retrieval_prefetch", action="store_true", help="Disable background prefetch of future retrieval descriptors.")
    parser.add_argument("--retrieval_prefetch_count", type=int, default=16, help="How many upcoming images to prefetch retrieval descriptors for.")
    parser.add_argument("--alignment_point_conf_threshold", type=float, default=0.5, help="Confidence threshold for compact point correspondences in point_sim3 alignment.")
    parser.add_argument("--alignment_grid_topk_per_cell", type=int, default=12, help="Keep at most this many high-confidence correspondences per image grid cell.")
    parser.add_argument("--alignment_grid_conf_min", type=float, default=0.5, help="Minimum sqrt(local_conf * global_conf) required before a point can be sampled.")
    parser.add_argument("--alignment_compact_max_points", type=int, default=5000, help="Maximum compact alignment reference points retained per frame.")
    parser.add_argument("--alignment_compact_grid_cell_size", type=int, default=None, help="Grid cell size for compact alignment sampling. Defaults to 32 pixels.")
    parser.add_argument("--disable_alignment_validation", action="store_true", help="Disable the post-alignment validation gate before fusing a batch into the global model.")
    parser.add_argument("--validation_translation_spacing_multiplier", type=float, default=50.0, help="Adaptive translation threshold multiplier applied to the dataset-local adjacent-pixel 3D spacing median.")
    parser.add_argument("--validation_spacing_estimation_samples", type=int, default=2048, help="How many adjacent-pixel 3D spacing samples to use when estimating the adaptive validation threshold.")
    parser.add_argument("--validation_spacing_estimation_max_frames", type=int, default=5, help="How many bootstrap frames to sample when estimating the adaptive validation threshold.")
    parser.add_argument("--validation_median_rotation_deg_threshold", type=float, default=10.0, help="Maximum allowed median rotation residual in degrees against historical reference poses.")
    parser.add_argument("--validation_max_rotation_deg_threshold", type=float, default=25.0, help="Maximum allowed worst-case rotation residual in degrees against historical reference poses.")
    parser.add_argument("--validation_scale_min", type=float, default=0.5, help="Minimum acceptable estimated Sim3 scale during validation.")
    parser.add_argument("--validation_scale_max", type=float, default=1.5, help="Maximum acceptable estimated Sim3 scale during validation.")
    parser.add_argument("--enable_register_ref_pruning_retry", action="store_true", help="After validation failure, first retry by geometry-based ref pruning on the existing subset, then fall back to VFM register-token ref pruning with one re-inference.")
    parser.add_argument("--register_ref_pruning_candidate_z", type=float, default=2.0, help="Robust-z threshold on VFM register-token distance to the new-frame anchor for retry ref pruning.")
    parser.add_argument("--register_ref_pruning_min_similarity", type=float, default=0.0, help="Drop retry reference candidates whose VFM register-token cosine similarity to the new-frame anchor is at or below this value.")
    parser.add_argument("--register_ref_pruning_max_drop_refs", type=int, default=1, help="Maximum number of reference frames dropped before the VFM register-token retry.")
    parser.add_argument("--register_ref_pruning_min_refs", type=int, default=0, help="Minimum references kept for the VFM register-token retry. Default 0 uses --min_topk_for_alignment.")
    parser.add_argument("--fusion_conf_threshold", type=float, default=0.5, help="Confidence threshold for fusing new-frame points.")
    parser.add_argument("--image_preprocess_cache_max_images", type=int, default=1500, help="Maximum number of CPU preprocessed image entries kept in the runtime LRU cache.")
    parser.add_argument("--image_preprocess_prefetch_count", type=int, default=20, help="How many upcoming images to keep ahead in the VFM image preprocess prefetch queue.")
    parser.add_argument("--image_preprocess_prefetch_workers", type=int, default=1, help="Number of CPU workers used for image preprocess prefetch.")
    parser.add_argument("--enable_pose_graph_optimization", action="store_true", help="Enable final frame-level SE3 pose-graph optimization. PGO updates camera poses only; fused point clouds are not transformed.")
    parser.add_argument("--pose_opt_min_edges", type=int, default=30, help="Minimum pose-graph edges required for final optimization.")
    parser.add_argument("--pose_opt_max_nfev", type=int, default=15, help="Maximum optimizer iterations/evaluations for one pose-graph job.")
    parser.add_argument("--save_pose_graph_debug", action="store_true", help="Export PGO nodes/edges CSV/JSON plus 2D/3D spatial graph PNGs.")
    parser.add_argument("--save_batch_logs", action="store_true", help="Save per-batch reconstruction logs to batch_logs.json for debugging and ablation analysis.")
    parser.add_argument("--save_ply", action="store_true", help="Also export the fused global point cloud as PLY.")
    parser.add_argument("--ply_export_mode", type=str, default="dedup", choices=["dedup", "raw"], help="PLY export mode: raw keeps all fused points; dedup uses Numba voxel deduplication.")
    parser.add_argument("--final_ply_dedup_voxel_multiplier", type=float, default=1.0, help="Voxel size multiplier for final dedup PLY: dataset_spacing_reference * multiplier.")
    parser.add_argument("--disable_pose_export", action="store_true", help="Do not export reconstructed camera poses after the run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    defaults = ReconstructionConfig()
    parser.set_defaults(
        model=defaults.model_name,
        init_window=defaults.init_window,
        max_batch_size=defaults.max_batch_size,
        dynamic_batch_ref_overlap_min=defaults.dynamic_batch_ref_overlap_min,
        dynamic_batch_ref_overlap_ratio_min=defaults.dynamic_batch_ref_overlap_ratio_min,
        dynamic_batch_query_similarity_min=defaults.dynamic_batch_query_similarity_min,
        dynamic_batch_dominant_refs=defaults.dynamic_batch_dominant_refs,
        retrieval_topk=defaults.retrieval_topk,
        min_topk_for_alignment=defaults.min_topk_for_alignment,
        use_sky_mask=defaults.use_sky_mask,
        supscene_weights=defaults.supscene_weights,
        supscene_image_size=defaults.supscene_image_size,
        enable_profiling_sync=defaults.enable_profiling_sync,
        enable_detailed_profiling=defaults.enable_detailed_profiling,
        disable_retrieval_prefetch=not defaults.enable_retrieval_prefetch,
        retrieval_prefetch_count=defaults.retrieval_prefetch_count,
        alignment_point_conf_threshold=defaults.alignment_point_conf_threshold,
        alignment_grid_topk_per_cell=defaults.alignment_grid_topk_per_cell,
        alignment_grid_conf_min=defaults.alignment_grid_conf_min,
        alignment_compact_max_points=defaults.alignment_compact_max_points,
        enable_register_ref_pruning_retry=defaults.enable_register_ref_pruning_retry,
        register_ref_pruning_candidate_z=defaults.register_ref_pruning_candidate_z,
        register_ref_pruning_min_similarity=defaults.register_ref_pruning_min_similarity,
        register_ref_pruning_max_drop_refs=defaults.register_ref_pruning_max_drop_refs,
        register_ref_pruning_min_refs=defaults.register_ref_pruning_min_refs,
        fusion_conf_threshold=defaults.fusion_conf_threshold,
        image_preprocess_cache_max_images=defaults.image_preprocess_cache_max_images,
        image_preprocess_prefetch_count=defaults.image_preprocess_prefetch_count,
        image_preprocess_prefetch_workers=defaults.image_preprocess_prefetch_workers,
        disable_alignment_validation=not defaults.enable_alignment_validation,
        validation_translation_spacing_multiplier=defaults.validation_translation_spacing_multiplier,
        validation_spacing_estimation_samples=defaults.validation_spacing_estimation_samples,
        validation_spacing_estimation_max_frames=defaults.validation_spacing_estimation_max_frames,
        validation_median_rotation_deg_threshold=defaults.validation_median_rotation_deg_threshold,
        validation_max_rotation_deg_threshold=defaults.validation_max_rotation_deg_threshold,
        validation_scale_min=defaults.validation_scale_min,
        validation_scale_max=defaults.validation_scale_max,
        enable_pose_graph_optimization=defaults.enable_pose_graph_optimization,
        pose_opt_min_edges=defaults.pose_opt_min_edges,
        pose_opt_max_nfev=defaults.pose_opt_max_nfev,
    )
    return parser


_CONFIG_SECTION_KEYS = {
    "defaults",
    "editable",
    "common",
    "ablations",
    "cases",
    "experiments",
    "default_case",
    "case",
    "description",
    "notes",
}

_CONFIG_METADATA_KEYS = {"description", "notes", "tags"}


def _load_config_file(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as f:
        if suffix == ".json":
            data = json.load(f)
        else:
            try:
                import yaml
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "YAML config files require PyYAML. Install requirements.txt or use a JSON config."
                ) from exc
            data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}.")
    return data


def _bool_from_config(value: object, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Config key '{key}' must be a boolean value, got {value!r}.")


def _parser_destinations(parser: argparse.ArgumentParser) -> set[str]:
    return {
        action.dest
        for action in parser._actions
        if action.dest not in {argparse.SUPPRESS, "help"}
    }


def _normalize_config_values(
    values: dict,
    *,
    valid_dests: set[str],
    source: str,
) -> dict:
    normalized: dict[str, object] = {}
    for key, value in values.items():
        if key in _CONFIG_METADATA_KEYS:
            continue
        if key in {"alignment_validation", "validation"}:
            normalized["disable_alignment_validation"] = not _bool_from_config(value, key=key)
        elif key == "validation_retry":
            enabled = _bool_from_config(value, key=key)
            normalized["disable_alignment_validation"] = not enabled
            normalized["enable_register_ref_pruning_retry"] = enabled
        elif key in {"pgo", "pose_graph_optimization"}:
            normalized["enable_pose_graph_optimization"] = _bool_from_config(value, key=key)
        elif key in valid_dests:
            normalized[key] = value
        else:
            raise ValueError(f"Unknown config key '{key}' in {source}.")
    return normalized


def _flat_top_level_config(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key not in _CONFIG_SECTION_KEYS
    }


def _resolve_config_defaults(
    *,
    config_path: str,
    requested_case: str | None,
    valid_dests: set[str],
) -> tuple[dict, dict]:
    config = _load_config_file(config_path)
    selected_case = requested_case or config.get("case") or config.get("default_case")
    selected_case = str(selected_case) if selected_case not in (None, "") else None

    merged: dict[str, object] = {}
    applied_layers: list[str] = []

    flat_top = _flat_top_level_config(config)
    if flat_top:
        merged.update(
            _normalize_config_values(
                flat_top,
                valid_dests=valid_dests,
                source=f"{config_path}:<top-level>",
            )
        )
        applied_layers.append("<top-level>")

    for section in ("defaults", "common", "editable"):
        section_values = config.get(section)
        if section_values is None:
            continue
        if not isinstance(section_values, dict):
            raise ValueError(f"Config section '{section}' must be a mapping.")
        merged.update(
            _normalize_config_values(
                section_values,
                valid_dests=valid_dests,
                source=f"{config_path}:{section}",
            )
        )
        applied_layers.append(section)

    case_source = None
    if selected_case is not None:
        for section in ("ablations", "cases", "experiments"):
            section_values = config.get(section)
            if section_values is None:
                continue
            if not isinstance(section_values, dict):
                raise ValueError(f"Config section '{section}' must be a mapping.")
            if selected_case in section_values:
                case_values = section_values[selected_case]
                if not isinstance(case_values, dict):
                    raise ValueError(f"Config case '{selected_case}' in '{section}' must be a mapping.")
                merged.update(
                    _normalize_config_values(
                        case_values,
                        valid_dests=valid_dests,
                        source=f"{config_path}:{section}.{selected_case}",
                    )
                )
                applied_layers.append(f"{section}.{selected_case}")
                case_source = section
                break
        if case_source is None:
            available_cases: list[str] = []
            for section in ("ablations", "cases", "experiments"):
                section_values = config.get(section)
                if isinstance(section_values, dict):
                    available_cases.extend([f"{section}.{name}" for name in section_values])
            raise ValueError(
                f"Config case '{selected_case}' was not found. "
                f"Available cases: {available_cases or 'none'}"
            )

    if selected_case is not None and not merged.get("output_dir") and merged.get("output_root"):
        merged["output_dir"] = str(Path(str(merged["output_root"])) / selected_case)
    if selected_case is not None:
        merged["config_case"] = selected_case
    merged["config"] = str(config_path)

    info = {
        "path": str(config_path),
        "case": selected_case,
        "case_source": case_source,
        "applied_layers": applied_layers,
        "derived_output_dir_from_output_root": bool(
            selected_case is not None
            and bool(merged.get("output_root"))
            and str(merged.get("output_dir", "")).endswith(f"/{selected_case}")
        ),
    }
    return merged, info


def parse_run_args(
    argv: list[str] | None = None,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> tuple[argparse.Namespace, dict]:
    argv = list(sys.argv[1:] if argv is None else argv)
    cli_output_dir_provided = any(
        item == "--output_dir" or item.startswith("--output_dir=")
        for item in argv
    )
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_parser.add_argument("--case", "--config_case", dest="config_case", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = parser or build_parser()
    config_info = {
        "path": None,
        "case": None,
        "case_source": None,
        "applied_layers": [],
        "derived_output_dir_from_output_root": False,
    }
    if pre_args.config:
        config_defaults, config_info = _resolve_config_defaults(
            config_path=pre_args.config,
            requested_case=pre_args.config_case,
            valid_dests=_parser_destinations(parser),
        )
        parser.set_defaults(**config_defaults)

    args = parser.parse_args(argv)
    if args.image_dir is None:
        parser.error("--image_dir is required unless it is provided by --config.")
    should_derive_case_output = (
        args.output_root
        and args.config_case
        and (
            args.output_dir is None
            or (
                config_info.get("derived_output_dir_from_output_root")
                and not cli_output_dir_provided
            )
        )
    )
    if should_derive_case_output:
        args.output_dir = str(Path(args.output_root) / args.config_case)
        config_info["derived_output_dir_from_output_root"] = True
    elif args.output_dir is None:
        args.output_dir = str(Path(args.output_root)) if args.output_root else "outputs/incremental_run"
    config_info["effective_output_dir"] = args.output_dir
    return args, config_info

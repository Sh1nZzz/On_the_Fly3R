"""Command-line entrypoint for progressive multi-VFM reconstruction."""

from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from on_the_fly3r import IncrementalReconstructor, ReconstructionConfig
from on_the_fly3r.config import parse_run_args
from on_the_fly3r.utils import iter_image_paths, set_seed
from on_the_fly3r.output import (
    _collect_summary_metrics,
    _write_pose_exports,
    write_batch_logs,
    write_profile_summary,
    write_resolved_config,
    write_run_summary,
)

def main() -> None:
    args, config_info = parse_run_args()
    set_seed(args.seed)
    image_paths = iter_image_paths(args.image_dir)
    if int(args.max_images) > 0:
        image_paths = image_paths[: int(args.max_images)]
    if len(image_paths) < args.init_window:
        raise ValueError(
            f"Found {len(image_paths)} images in {args.image_dir}, "
            f"but init_window={args.init_window}."
        )

    output_dir = Path(args.output_dir)
    resolved_config_path = write_resolved_config(output_dir, args, config_info)
    print(f"Saved resolved config to {resolved_config_path}")

    config = ReconstructionConfig(
        model_name=args.model,
        init_window=args.init_window,
        max_batch_size=args.max_batch_size,
        dynamic_batch_ref_overlap_min=args.dynamic_batch_ref_overlap_min,
        dynamic_batch_ref_overlap_ratio_min=args.dynamic_batch_ref_overlap_ratio_min,
        dynamic_batch_query_similarity_min=args.dynamic_batch_query_similarity_min,
        dynamic_batch_dominant_refs=args.dynamic_batch_dominant_refs,
        retrieval_topk=args.retrieval_topk,
        min_topk_for_alignment=args.min_topk_for_alignment,
        use_sky_mask=args.use_sky_mask,
        inference_device=args.inference_device,
        retrieval_device=args.retrieval_device,
        supscene_weights=args.supscene_weights,
        supscene_image_size=args.supscene_image_size,
        enable_profiling_sync=args.enable_profiling_sync,
        enable_detailed_profiling=args.enable_detailed_profiling,
        enable_retrieval_prefetch=not args.disable_retrieval_prefetch,
        retrieval_prefetch_count=args.retrieval_prefetch_count,
        alignment_point_conf_threshold=args.alignment_point_conf_threshold,
        alignment_grid_topk_per_cell=args.alignment_grid_topk_per_cell,
        alignment_grid_conf_min=args.alignment_grid_conf_min,
        alignment_compact_max_points=args.alignment_compact_max_points,
        alignment_compact_grid_cell_size=args.alignment_compact_grid_cell_size,
        enable_alignment_validation=not args.disable_alignment_validation,
        validation_translation_spacing_multiplier=args.validation_translation_spacing_multiplier,
        validation_spacing_estimation_samples=args.validation_spacing_estimation_samples,
        validation_spacing_estimation_max_frames=args.validation_spacing_estimation_max_frames,
        validation_median_rotation_deg_threshold=args.validation_median_rotation_deg_threshold,
        validation_max_rotation_deg_threshold=args.validation_max_rotation_deg_threshold,
        validation_scale_min=args.validation_scale_min,
        validation_scale_max=args.validation_scale_max,
        enable_register_ref_pruning_retry=args.enable_register_ref_pruning_retry,
        register_ref_pruning_candidate_z=args.register_ref_pruning_candidate_z,
        register_ref_pruning_min_similarity=args.register_ref_pruning_min_similarity,
        register_ref_pruning_max_drop_refs=args.register_ref_pruning_max_drop_refs,
        register_ref_pruning_min_refs=args.register_ref_pruning_min_refs,
        fusion_conf_threshold=args.fusion_conf_threshold,
        image_preprocess_cache_max_images=args.image_preprocess_cache_max_images,
        image_preprocess_prefetch_count=args.image_preprocess_prefetch_count,
        image_preprocess_prefetch_workers=args.image_preprocess_prefetch_workers,
        enable_pose_graph_optimization=args.enable_pose_graph_optimization,
        pose_opt_min_edges=args.pose_opt_min_edges,
        pose_opt_max_nfev=args.pose_opt_max_nfev,
    )

    model_args = SimpleNamespace(model=args.model, model_checkpoint=args.model_checkpoint)
    reconstructor = IncrementalReconstructor(config=config, model_args=model_args)

    print(f"Found {len(image_paths)} images in {args.image_dir}")
    print(f"Bootstrapping with first {args.init_window} images")
    print(f"Retrieval backend: SupScene with Torch GPU search on {args.retrieval_device}")
    print("Online fusion dedup: disabled")
    print("Alignment sampling mode: grid_conf_topk")
    print("Retain frame images: False")
    print("Retain frame dense maps: False (compact reference only)")
    print("Image preprocess cache: cpu")
    print(
        "Image preprocess prefetch: enabled "
        f"(count={config.image_preprocess_prefetch_count}, "
        f"workers={config.image_preprocess_prefetch_workers})"
    )
    if args.enable_detailed_profiling:
        print("Detailed profiling: enabled")
    if args.enable_pose_graph_optimization:
        print(
            "Pose graph optimization: enabled "
            f"(mode=final, min_edges={args.pose_opt_min_edges}, "
            f"max_nfev={args.pose_opt_max_nfev})"
        )
    else:
        print("Pose graph optimization: disabled")
    remaining_images = max(0, len(image_paths) - args.init_window)
    progress_bar = tqdm(total=remaining_images, desc="Incremental Images", unit="image") if remaining_images > 0 else None

    def on_batch_end(batch_idx: int, _total: int, log: dict) -> None:
        if progress_bar is None:
            return
        status = log.get("status", "unknown")
        retrieval_time = log.get("retrieval", {}).get("time_sec", 0.0)
        inference_time = log.get("inference", {}).get("time_sec", 0.0)
        frame_ids = log.get("frame_ids", [])
        frame_label = frame_ids[0] if len(frame_ids) == 1 else f"{len(frame_ids)} frames"
        progress_bar.set_postfix_str(
            f"batch={batch_idx} status={status} item={frame_label} ret={retrieval_time:.2f}s inf={inference_time:.2f}s"
        )
        progress_bar.update(len(frame_ids))

    logs = []
    runtime_summary = {"bootstrap": {}, "bootstrap_time_sec": 0.0, "processing_time_sec": 0.0, "total_time_sec": 0.0}
    pose_export_summary = None
    ply_export_summary = None
    pose_graph_debug_summary = None
    try:
        logs = reconstructor.run_dataset(image_paths, batch_callback=on_batch_end)
        reconstructor.finalize_pose_graph_optimization()
        if args.save_pose_graph_debug:
            pose_graph_debug_summary = reconstructor.export_pose_graph_debug(output_dir / "pose_graph_debug")
            print(f"Saved pose graph debug to {pose_graph_debug_summary['debug_dir']}")
        runtime_summary = reconstructor.get_runtime_summary()

        if not args.disable_pose_export:
            pose_export_summary = _write_pose_exports(
                reconstructor=reconstructor,
                image_paths=image_paths,
                output_dir=output_dir,
            )
            print(f"Saved camera poses to {pose_export_summary['camera_poses_npz']}")

        if args.save_ply:
            ply_export_summary = {
                "mode": args.ply_export_mode,
                "final_ply_dedup_voxel_multiplier": args.final_ply_dedup_voxel_multiplier,
                "exports": [],
            }
            if args.ply_export_mode == "raw":
                ply_path = output_dir / "global_points.ply"
                stats = reconstructor.export_global_ply(str(ply_path), mode="raw")
                ply_export_summary["exports"].append(stats)
                print(f"Saved raw fused point cloud to {ply_path}")
            if args.ply_export_mode == "dedup":
                ply_path = output_dir / "global_points.ply"
                stats = reconstructor.export_global_ply(
                    str(ply_path),
                    mode="dedup",
                    final_dedup_voxel_multiplier=args.final_ply_dedup_voxel_multiplier,
                )
                ply_export_summary["exports"].append(stats)
                print(
                    f"Saved dedup point cloud to {ply_path} "
                    f"({stats.get('exported_vertex_count')} / {stats.get('raw_vertex_count')} vertices)"
                )
    finally:
        reconstructor.shutdown()
        if progress_bar is not None:
            progress_bar.close()

    summary = {
        "image_dir": args.image_dir,
        "output_dir": str(output_dir),
        "output_root": args.output_root,
        "config_path": args.config,
        "config_case": args.config_case,
        "resolved_config": str(resolved_config_path),
        "num_images": len(image_paths),
        "max_images": int(args.max_images),
        "model": args.model,
        "init_window": args.init_window,
        "max_batch_size": args.max_batch_size,
        "dynamic_batching": True,
        "retrieval_backend": "supscene",
        "retrieval_device": args.retrieval_device,
        "detailed_profiling_enabled": args.enable_detailed_profiling,
        "retain_frame_images": False,
        "retain_frame_dense_maps": False,
        "supscene_weights": args.supscene_weights,
        "supscene_image_size": args.supscene_image_size,
        "alignment_mode": "point_sim3",
        "dataset_spacing_reference": runtime_summary.get("bootstrap", {}).get("dataset_spacing_reference"),
        "validation_translation_threshold": runtime_summary.get("bootstrap", {}).get("validation_translation_threshold"),
        "register_ref_pruning_retry_enabled": args.enable_register_ref_pruning_retry,
        "alignment_sampling_mode": "grid_conf_topk",
        "alignment_grid_topk_per_cell": args.alignment_grid_topk_per_cell,
        "alignment_grid_conf_min": args.alignment_grid_conf_min,
        "alignment_reference_mode": "compact",
        "alignment_compact_max_points": args.alignment_compact_max_points,
        "alignment_compact_grid_cell_size": args.alignment_compact_grid_cell_size,
        "fusion_dedup_enabled": False,
        "fusion_dedup_mode": "off",
        "image_preprocess_cache": "cpu",
        "image_preprocess_cache_max_images": args.image_preprocess_cache_max_images,
        "image_preprocess_cache_stats": runtime_summary.get("image_preprocess_cache"),
        "image_preprocess_prefetch_enabled": True,
        "image_preprocess_prefetch_count": config.image_preprocess_prefetch_count,
        "image_preprocess_prefetch_workers": config.image_preprocess_prefetch_workers,
        "image_preprocess_prefetch_stats": runtime_summary.get("image_preprocess_prefetch"),
        "pose_graph_optimization_enabled": args.enable_pose_graph_optimization,
        "pose_graph_mode": "final",
        "pose_opt_min_edges": args.pose_opt_min_edges,
        "pose_opt_max_nfev": args.pose_opt_max_nfev,
        "save_pose_graph_debug": args.save_pose_graph_debug,
        "num_batches": len(logs),
        "num_added_batches": sum(1 for item in logs if item.get("status") == "added"),
        "num_skipped_batches": sum(1 for item in logs if item.get("status") != "added"),
        "num_added_frames": sum(len(item.get("frame_ids", [])) for item in logs if item.get("status") == "added"),
        "runtime": runtime_summary,
        "pose_graph": runtime_summary.get("pose_graph"),
        "pose_graph_debug": pose_graph_debug_summary,
        "runtime_averages": _collect_summary_metrics(logs, runtime_summary),
        "pose_export": pose_export_summary,
        "ply_export": ply_export_summary,
    }
    if args.enable_detailed_profiling:
        profile_summary_path = write_profile_summary(output_dir, logs, runtime_summary)
        summary["profile_summary"] = str(profile_summary_path)
        print(f"Saved profile summary to {profile_summary_path}")
    if args.save_batch_logs:
        batch_logs_path = write_batch_logs(output_dir, logs)
        summary["batch_logs"] = str(batch_logs_path)
        print(f"Saved batch logs to {batch_logs_path}")
    summary_path = write_run_summary(output_dir, summary)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

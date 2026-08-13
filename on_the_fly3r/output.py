"""Stable writers for run metadata, summaries, profiles, and poses."""

import argparse
import csv
import datetime as _datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    return path


def write_resolved_config(
    output_dir: Path,
    args: argparse.Namespace,
    config_info: dict,
) -> Path:
    return write_json(
        output_dir / "resolved_config.json",
        _build_resolved_config_document(args, config_info),
    )


def write_profile_summary(output_dir: Path, logs: list[dict], runtime_summary: dict) -> Path:
    return write_json(
        output_dir / "profile_summary.json",
        _collect_profile_summary(logs, runtime_summary),
    )


def write_batch_logs(output_dir: Path, logs: list[dict]) -> Path:
    return write_json(output_dir / "batch_logs.json", _json_ready(logs))


def write_run_summary(output_dir: Path, summary: dict) -> Path:
    return write_json(output_dir / "summary.json", summary)


def _run_git_command(repo_dir: Path, command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["git", *command],
            cwd=str(repo_dir),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return 1, str(exc)
    return int(result.returncode), result.stdout.strip()


def _collect_git_metadata() -> dict:
    repo_dir = Path(__file__).resolve().parent
    commit_code, commit = _run_git_command(repo_dir, ["rev-parse", "HEAD"])
    branch_code, branch = _run_git_command(repo_dir, ["branch", "--show-current"])
    status_code, status = _run_git_command(repo_dir, ["status", "--short"])
    return {
        "commit": commit if commit_code == 0 else None,
        "branch": branch if branch_code == 0 else None,
        "dirty": bool(status) if status_code == 0 else None,
        "status_short": status.splitlines() if status_code == 0 and status else [],
    }


def _namespace_to_dict(args: argparse.Namespace) -> dict:
    return {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_resolved_config_document(args: argparse.Namespace, config_info: dict) -> dict:
    args_dict = _namespace_to_dict(args)
    return {
        "metadata": {
            "created_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "cwd": str(Path.cwd()),
            "command": shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]]),
            "python": sys.version,
            "torch_version": getattr(torch, "__version__", None),
            "git": _collect_git_metadata(),
        },
        "config": config_info,
        "args": args_dict,
        "derived": {
            "dynamic_batching": True,
            "alignment_validation": not bool(args.disable_alignment_validation),
            "validation_retry": (
                not bool(args.disable_alignment_validation)
                and bool(args.enable_register_ref_pruning_retry)
            ),
            "pgo": bool(args.enable_pose_graph_optimization),
            "effective_ply_outputs": ["global_points.ply"] if args.save_ply else [],
        },
    }


def _mean_or_zero(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
    )


def _add_profile_values(groups: dict[str, list[float]], prefix: str, values: dict | None) -> None:
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        if _is_finite_number(value):
            groups.setdefault(f"{prefix}.{key}", []).append(float(value))


def _summarize_profile_groups(groups: dict[str, list[float]], prefix: str) -> dict:
    prefix_with_dot = f"{prefix}."
    summary = {}
    for key, values in sorted(groups.items()):
        if not key.startswith(prefix_with_dot) or not values:
            continue
        local_key = key[len(prefix_with_dot):]
        values_np = np.asarray(values, dtype=np.float64)
        summary[local_key] = {
            "count": int(values_np.size),
            "mean": float(np.mean(values_np)),
            "min": float(np.min(values_np)),
            "max": float(np.max(values_np)),
            "total": float(np.sum(values_np)),
        }
    return summary


def _collect_profile_summary(logs: list[dict], runtime_summary: dict) -> dict:
    groups: dict[str, list[float]] = {}
    slow_batches = []

    bootstrap = runtime_summary.get("bootstrap", {})
    if isinstance(bootstrap, dict):
        _add_profile_values(
            groups,
            "bootstrap.inference",
            bootstrap.get("inference", {}).get("profile", {}),
        )
        _add_profile_values(
            groups,
            "bootstrap.stage",
            {
                "inference_time_sec": bootstrap.get("inference", {}).get("time_sec"),
                "retrieval_time_sec": bootstrap.get("retrieval", {}).get("time_sec"),
                "fusion_time_sec": bootstrap.get("fusion_time_sec"),
                "total_time_sec": bootstrap.get("total_time_sec"),
            },
        )

    for batch_idx, log in enumerate(logs, start=1):
        _add_profile_values(
            groups,
            "batch.stage",
            {
                "retrieval_time_sec": log.get("retrieval", {}).get("time_sec"),
                "inference_time_sec": log.get("inference", {}).get("time_sec"),
                "alignment_time_sec": log.get("alignment_time_sec"),
                "validation_time_sec": log.get("validation_time_sec"),
                "fusion_time_sec": log.get("fusion_time_sec"),
                "total_time_sec": log.get("total_time_sec"),
                "batch_size": log.get("batch_size"),
            },
        )
        _add_profile_values(groups, "inference", log.get("inference", {}).get("profile", {}))
        _add_profile_values(groups, "alignment", log.get("diagnostics", {}).get("profile", {}))
        _add_profile_values(groups, "alignment.sampling", log.get("diagnostics", {}).get("sampling", {}))
        fusion_profile = log.get("fusion_profile")
        if isinstance(fusion_profile, dict):
            _add_profile_values(
                groups,
                "fusion.batch",
                {key: value for key, value in fusion_profile.items() if key != "frames"},
            )
            for frame_profile in fusion_profile.get("frames", []):
                if not isinstance(frame_profile, dict):
                    continue
                _add_profile_values(
                    groups,
                    "fusion.frame",
                    {
                        key: value
                        for key, value in frame_profile.items()
                        if key not in {"frame_id", "dedup_profile", "build_profile"}
                    },
                )
                _add_profile_values(groups, "fusion.frame.dedup", frame_profile.get("dedup_profile", {}))
                _add_profile_values(groups, "fusion.frame.build", frame_profile.get("build_profile", {}))

        total_time = log.get("total_time_sec")
        if _is_finite_number(total_time):
            slow_batches.append(
                {
                    "batch_idx": int(batch_idx),
                    "frame_ids": list(log.get("frame_ids", [])),
                    "status": log.get("status"),
                    "batch_size": int(log.get("batch_size") or len(log.get("frame_ids", []))),
                    "total_time_sec": float(total_time),
                    "retrieval_time_sec": float(log.get("retrieval", {}).get("time_sec") or 0.0),
                    "inference_time_sec": float(log.get("inference", {}).get("time_sec") or 0.0),
                    "alignment_time_sec": float(log.get("alignment_time_sec") or 0.0),
                    "validation_time_sec": float(log.get("validation_time_sec") or 0.0),
                    "fusion_time_sec": float(log.get("fusion_time_sec") or 0.0),
                }
            )

    slow_batches.sort(key=lambda item: item["total_time_sec"], reverse=True)
    image_cache_stats = runtime_summary.get("image_preprocess_cache", {})
    image_prefetch_stats = runtime_summary.get("image_preprocess_prefetch", {})
    return {
        "num_profiled_batches": int(len(logs)),
        "bootstrap_stage": _summarize_profile_groups(groups, "bootstrap.stage"),
        "bootstrap_inference_profile": _summarize_profile_groups(groups, "bootstrap.inference"),
        "batch_stage": _summarize_profile_groups(groups, "batch.stage"),
        "inference_profile": _summarize_profile_groups(groups, "inference"),
        "alignment_profile": _summarize_profile_groups(groups, "alignment"),
        "alignment_sampling_profile": _summarize_profile_groups(groups, "alignment.sampling"),
        "fusion_batch_profile": _summarize_profile_groups(groups, "fusion.batch"),
        "fusion_frame_profile": _summarize_profile_groups(groups, "fusion.frame"),
        "fusion_frame_dedup_profile": _summarize_profile_groups(groups, "fusion.frame.dedup"),
        "fusion_frame_build_profile": _summarize_profile_groups(groups, "fusion.frame.build"),
        "image_preprocess_cache": image_cache_stats if isinstance(image_cache_stats, dict) else {},
        "image_preprocess_prefetch": image_prefetch_stats if isinstance(image_prefetch_stats, dict) else {},
        "slowest_batches": slow_batches[:10],
    }


def _collect_summary_metrics(logs: list[dict], runtime_summary: dict) -> dict:
    retrieval_times = [item.get("retrieval", {}).get("time_sec") for item in logs if item.get("retrieval") is not None]
    inference_times = [item.get("inference", {}).get("time_sec") for item in logs if item.get("inference") is not None]
    alignment_times = [item.get("alignment_time_sec") for item in logs if item.get("alignment_time_sec") is not None]
    fusion_times = [item.get("fusion_time_sec") for item in logs if item.get("fusion_time_sec") is not None]
    batch_total_times = [item.get("total_time_sec") for item in logs if item.get("total_time_sec") is not None]

    point_sim3_logs = [
        item for item in logs
        if item.get("diagnostics", {}).get("mode") == "point_sim3"
    ]
    validation_logs = [item for item in logs if item.get("validation") is not None]
    pose_graph_events = list(runtime_summary.get("pose_graph", {}).get("events", []))
    if not pose_graph_events:
        pose_graph_events = [
            event
            for item in logs
            for event in item.get("pose_graph_events", [])
        ]
    graph_events = pose_graph_events
    merged_pose_graph_events = [
        event
        for event in graph_events
        if event.get("event") == "pose_graph_merged"
    ]
    final_pose_graph_events = [event for event in merged_pose_graph_events if event.get("phase") == "final"]
    failed_pose_graph_events = [
        event
        for event in graph_events
        if event.get("event") == "pose_graph_failed"
    ]
    skipped_final_pose_graph_events = [
        event
        for event in graph_events
        if event.get("event") == "pose_graph_final_skipped"
    ]
    num_added_frames = sum(len(item.get("frame_ids", [])) for item in logs if item.get("status") == "added")

    return {
        "mean_retrieval_time_sec": _mean_or_zero(retrieval_times),
        "mean_inference_time_sec": _mean_or_zero(inference_times),
        "mean_alignment_time_sec": _mean_or_zero(alignment_times),
        "mean_fusion_time_sec": _mean_or_zero(fusion_times),
        "mean_total_time_sec_per_batch": _mean_or_zero(batch_total_times),
        "mean_total_time_sec_per_added_frame": (
            float(runtime_summary.get("processing_time_sec", 0.0)) / float(num_added_frames)
            if num_added_frames > 0
            else 0.0
        ),
        "mean_point_sim3_num_correspondences": _mean_or_zero(
            [item.get("diagnostics", {}).get("num_correspondences") for item in point_sim3_logs]
        ),
        "mean_point_sim3_num_inliers": _mean_or_zero(
            [item.get("diagnostics", {}).get("num_inliers") for item in point_sim3_logs]
        ),
        "mean_point_sim3_median_point_residual": _mean_or_zero(
            [item.get("diagnostics", {}).get("median_point_residual") for item in point_sim3_logs]
        ),
        "mean_point_sim3_max_point_residual": _mean_or_zero(
            [item.get("diagnostics", {}).get("max_point_residual") for item in point_sim3_logs]
        ),
        "mean_alignment_sampling_candidates": _mean_or_zero(
            [item.get("diagnostics", {}).get("sampling", {}).get("candidate_count") for item in point_sim3_logs]
        ),
        "mean_alignment_sampling_selected": _mean_or_zero(
            [item.get("diagnostics", {}).get("sampling", {}).get("selected_count") for item in point_sim3_logs]
        ),
        "mean_validation_median_translation_residual": _mean_or_zero(
            [item.get("validation", {}).get("metrics", {}).get("median_translation_residual") for item in validation_logs]
        ),
        "mean_validation_max_translation_residual": _mean_or_zero(
            [item.get("validation", {}).get("metrics", {}).get("max_translation_residual") for item in validation_logs]
        ),
        "mean_validation_median_rotation_residual_deg": _mean_or_zero(
            [item.get("validation", {}).get("metrics", {}).get("median_rotation_residual_deg") for item in validation_logs]
        ),
        "mean_validation_max_rotation_residual_deg": _mean_or_zero(
            [item.get("validation", {}).get("metrics", {}).get("max_rotation_residual_deg") for item in validation_logs]
        ),
        "num_pose_graph_events": int(len(graph_events)),
        "num_pose_graph_merges": int(len(merged_pose_graph_events)),
        "num_final_pose_graph_merges": int(len(final_pose_graph_events)),
        "num_pose_graph_failures": int(len(failed_pose_graph_events)),
        "num_final_pose_graph_skips": int(len(skipped_final_pose_graph_events)),
        "total_pose_graph_pgo_time_sec": float(sum(
            float(item.get("pgo_time_sec") or 0.0)
            for item in merged_pose_graph_events
        )),
        "total_pose_graph_pose_update_time_sec": float(sum(
            float(item.get("pose_update_time_sec") or 0.0)
            for item in merged_pose_graph_events
        )),
        "mean_pose_graph_pgo_time_sec": _mean_or_zero(
            [item.get("pgo_time_sec") for item in merged_pose_graph_events]
        ),
        "mean_pose_graph_pose_update_time_sec": _mean_or_zero(
            [item.get("pose_update_time_sec") for item in merged_pose_graph_events]
        ),
        "final_pose_graph_total_time_sec": _mean_or_zero(
            [item.get("final_total_time_sec") for item in final_pose_graph_events]
        ),
    }



def _write_pose_exports(
    reconstructor: Any,
    image_paths: list[str],
    output_dir: Path,
) -> dict:
    pose_data = reconstructor.export_camera_poses(image_paths=image_paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "camera_poses.npz"
    np.savez_compressed(
        npz_path,
        frame_ids=pose_data["frame_ids"],
        image_paths=pose_data["image_paths"],
        cam2world=pose_data["cam2world"],
        intrinsic=pose_data["intrinsic"],
        valid=pose_data["valid"],
    )

    missing_frame_ids = [
        str(frame_id)
        for frame_id, valid in zip(pose_data["frame_ids"].tolist(), pose_data["valid"].tolist())
        if not valid
    ]

    valid_count = int(np.count_nonzero(pose_data["valid"]))
    export_summary = {
        "camera_poses_npz": str(npz_path),
        "num_requested_frames": int(len(pose_data["frame_ids"])),
        "num_valid_frames": int(valid_count),
        "num_missing_frames": int(len(missing_frame_ids)),
        "missing_frame_ids": missing_frame_ids,
    }
    summary_path = output_dir / "pose_export_summary.json"
    write_json(summary_path, export_summary)
    export_summary["pose_export_summary"] = str(summary_path)
    return export_summary


class PoseGraphOutputMixin:
    def export_pose_graph_debug(self, debug_dir) -> Dict[str, object]:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        payload = self._last_pose_graph_debug_payload or {
            "frame_ids": self.state.ordered_frame_ids(),
            "initial_centers": {},
            "refined_centers": {},
            "summary": {},
        }
        initial_centers = dict(payload.get("initial_centers", {}))
        refined_centers = dict(payload.get("refined_centers", {}))
        optimized_ids = set(str(frame_id) for frame_id in payload.get("frame_ids", []))

        node_rows: List[Dict[str, object]] = []
        for frame_id in self.state.ordered_frame_ids():
            frame = self.state.get_frame(frame_id)
            current_center = np.asarray(frame.cam2world, dtype=np.float64)[:3, 3]
            before = np.asarray(initial_centers.get(frame_id, current_center), dtype=np.float64)
            after = np.asarray(refined_centers.get(frame_id, current_center), dtype=np.float64)
            order_index = self.state.get_frame_order_index(frame_id)
            node_rows.append(
                {
                    "frame_id": frame_id,
                    "image_path": frame.image_path,
                    "order_index": int(order_index) if order_index is not None else -1,
                    "optimized": frame_id in optimized_ids,
                    "before_x": float(before[0]),
                    "before_y": float(before[1]),
                    "before_z": float(before[2]),
                    "after_x": float(after[0]),
                    "after_y": float(after[1]),
                    "after_z": float(after[2]),
                }
            )

        edge_rows: List[Dict[str, object]] = []
        edge_type_counts: Dict[str, int] = {}
        for edge in self.pose_graph_edges:
            source_order = self.state.get_frame_order_index(edge.source_id)
            target_order = self.state.get_frame_order_index(edge.target_id)
            frame_distance = (
                abs(int(target_order) - int(source_order))
                if source_order is not None and target_order is not None
                else None
            )
            edge_type = str(edge.edge_type)
            edge_type_counts[edge_type] = int(edge_type_counts.get(edge_type, 0)) + 1
            edge_rows.append(
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "edge_type": edge_type,
                    "weight": float(edge.weight),
                    "frame_distance": int(frame_distance) if frame_distance is not None else "",
                    "is_loop": bool(edge_type == "loop" or (frame_distance is not None and frame_distance >= int(self.config.pose_graph_loop_min_separation))),
                }
            )

        nodes_csv = debug_path / "nodes.csv"
        edges_csv = debug_path / "edges.csv"
        counts_json = debug_path / "edge_type_counts.json"
        graph_json = debug_path / "graph.json"
        self._write_csv(nodes_csv, node_rows)
        self._write_csv(edges_csv, edge_rows)
        counts_payload = {
            "edge_type_counts": edge_type_counts,
            "num_nodes": int(len(node_rows)),
            "num_edges": int(len(edge_rows)),
        }
        with counts_json.open("w", encoding="utf-8") as f:
            json.dump(counts_payload, f, indent=2, ensure_ascii=False)
        graph_payload = {
            "nodes": node_rows,
            "edges": edge_rows,
            "pose_graph_summary": payload.get("summary", {}),
            **counts_payload,
        }
        with graph_json.open("w", encoding="utf-8") as f:
            json.dump(graph_payload, f, indent=2, ensure_ascii=False)

        png_outputs: Dict[str, str] = {}
        plot_error = None
        try:
            png_outputs = self._write_pose_graph_plots(debug_path, node_rows, edge_rows)
        except Exception as exc:
            plot_error = str(exc)

        summary = {
            "debug_dir": str(debug_path),
            "nodes_csv": str(nodes_csv),
            "edges_csv": str(edges_csv),
            "graph_json": str(graph_json),
            "edge_type_counts_json": str(counts_json),
            "plots": png_outputs,
            "plot_error": plot_error,
            **counts_payload,
        }
        summary_path = debug_path / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        summary["summary_json"] = str(summary_path)
        return summary

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
        fieldnames: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_pose_graph_plots(
        self,
        debug_path: Path,
        node_rows: Sequence[Dict[str, object]],
        edge_rows: Sequence[Dict[str, object]],
        *,
        prefix: str = "pose_graph",
        title_prefix: str = "Pose Graph",
    ) -> Dict[str, str]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        outputs: Dict[str, str] = {}
        before = {
            str(row["frame_id"]): np.asarray([row["before_x"], row["before_y"], row["before_z"]], dtype=np.float64)
            for row in node_rows
        }
        after = {
            str(row["frame_id"]): np.asarray([row["after_x"], row["after_y"], row["after_z"]], dtype=np.float64)
            for row in node_rows
        }
        ordered_ids = [str(row["frame_id"]) for row in sorted(node_rows, key=lambda row: int(row["order_index"]))]
        plot_edges = self._select_pose_graph_plot_edges(edge_rows)
        outputs[f"{prefix}_3d_before"] = str(debug_path / f"{prefix}_3d_before.png")
        outputs[f"{prefix}_3d_after"] = str(debug_path / f"{prefix}_3d_after.png")
        outputs[f"{prefix}_2d_xy_before"] = str(debug_path / f"{prefix}_2d_xy_before.png")
        outputs[f"{prefix}_2d_xy_after"] = str(debug_path / f"{prefix}_2d_xy_after.png")
        self._plot_pose_graph_3d(
            path=Path(outputs[f"{prefix}_3d_before"]),
            centers=before,
            ordered_ids=ordered_ids,
            edge_rows=plot_edges,
            title=f"{title_prefix} Before PGO",
            plt=plt,
        )
        self._plot_pose_graph_3d(
            path=Path(outputs[f"{prefix}_3d_after"]),
            centers=after,
            ordered_ids=ordered_ids,
            edge_rows=plot_edges,
            title=f"{title_prefix} After PGO",
            plt=plt,
        )
        self._plot_pose_graph_2d(
            path=Path(outputs[f"{prefix}_2d_xy_before"]),
            centers=before,
            ordered_ids=ordered_ids,
            edge_rows=plot_edges,
            title=f"{title_prefix} XY Before PGO",
            plt=plt,
        )
        self._plot_pose_graph_2d(
            path=Path(outputs[f"{prefix}_2d_xy_after"]),
            centers=after,
            ordered_ids=ordered_ids,
            edge_rows=plot_edges,
            title=f"{title_prefix} XY After PGO",
            plt=plt,
        )
        plt.close("all")
        return outputs

    @staticmethod
    def _select_pose_graph_plot_edges(edge_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        priority: List[Dict[str, object]] = []
        local: List[Dict[str, object]] = []
        for row in edge_rows:
            edge_type = str(row.get("edge_type", ""))
            if edge_type == "loop":
                priority.append(dict(row))
            else:
                local.append(dict(row))
        max_local = 1500
        if len(local) > max_local:
            stride = max(1, (len(local) + max_local - 1) // max_local)
            local = local[::stride]
        return local + priority

    @staticmethod
    def _edge_plot_style(row: Dict[str, object]) -> Dict[str, object]:
        edge_type = str(row.get("edge_type", ""))
        if edge_type == "loop":
            return {"color": "#d62728", "linewidth": 1.4, "alpha": 0.85, "linestyle": "-"}
        if edge_type == "local_continuity":
            return {"color": "#777777", "linewidth": 0.6, "alpha": 0.35, "linestyle": "-"}
        return {"color": "#aaaaaa", "linewidth": 0.45, "alpha": 0.18, "linestyle": "-"}

    @classmethod
    def _plot_pose_graph_3d(
        cls,
        *,
        path: Path,
        centers: Dict[str, np.ndarray],
        ordered_ids: Sequence[str],
        edge_rows: Sequence[Dict[str, object]],
        title: str,
        plt,
    ) -> None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        coords = np.asarray([centers[frame_id] for frame_id in ordered_ids if frame_id in centers], dtype=np.float64)
        if coords.size:
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color="#222222", linewidth=0.9, alpha=0.8)
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=8, color="#111111", alpha=0.9)
        for row in edge_rows:
            source = str(row.get("source", ""))
            target = str(row.get("target", ""))
            if source not in centers or target not in centers:
                continue
            a = centers[source]
            b = centers[target]
            style = cls._edge_plot_style(row)
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], **style)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        cls._set_3d_equalish(ax, coords)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    @classmethod
    def _plot_pose_graph_2d(
        cls,
        *,
        path: Path,
        centers: Dict[str, np.ndarray],
        ordered_ids: Sequence[str],
        edge_rows: Sequence[Dict[str, object]],
        title: str,
        plt,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        coords = np.asarray([centers[frame_id] for frame_id in ordered_ids if frame_id in centers], dtype=np.float64)
        if coords.size:
            ax.plot(coords[:, 0], coords[:, 1], color="#222222", linewidth=0.9, alpha=0.8)
            ax.scatter(coords[:, 0], coords[:, 1], s=8, color="#111111", alpha=0.9)
        for row in edge_rows:
            source = str(row.get("source", ""))
            target = str(row.get("target", ""))
            if source not in centers or target not in centers:
                continue
            a = centers[source]
            b = centers[target]
            style = cls._edge_plot_style(row)
            ax.plot([a[0], b[0]], [a[1], b[1]], **style)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        ax.grid(True, color="#dddddd", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    @staticmethod
    def _set_3d_equalish(ax, coords: np.ndarray) -> None:
        if coords.size == 0:
            return
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        center = (mins + maxs) * 0.5
        radius = float(np.max(maxs - mins) * 0.55)
        if not np.isfinite(radius) or radius <= 1e-9:
            radius = 1.0
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

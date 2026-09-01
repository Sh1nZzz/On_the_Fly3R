from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from on_the_fly3r.pose_graph import PoseGraphEdge, PoseGraphSnapshot, optimize_pose_graph
from on_the_fly3r import IncrementalReconstructor, ReconstructionConfig
from on_the_fly3r.config import build_parser, parse_run_args
from on_the_fly3r.retrieval import FrameIndex
from on_the_fly3r.pointcloud import (
    _write_binary_ply_streaming,
    _write_binary_ply_voxel_dedup_streaming,
)
from third_party_codes.vggt_long_code import robust_weighted_estimate_sim3


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = x
    return pose


def _ply_vertex_count(path: Path) -> int:
    with path.open("rb") as stream:
        for line in stream:
            text = line.decode("ascii")
            if text.startswith("element vertex "):
                return int(text.split()[-1])
            if text == "end_header\n":
                break
    raise AssertionError("PLY vertex count not found")


def test_cli_and_api_defaults_match_hav_ab05() -> None:
    args = build_parser().parse_args(["--image_dir", "/tmp/unused"])
    config = ReconstructionConfig()
    expected = (5, 5, 3, 1500)
    assert (
        args.max_batch_size,
        args.retrieval_topk,
        args.min_topk_for_alignment,
        args.image_preprocess_cache_max_images,
    ) == expected
    assert (
        config.max_batch_size,
        config.retrieval_topk,
        config.min_topk_for_alignment,
        config.image_preprocess_cache_max_images,
    ) == expected


def test_public_cli_excludes_removed_modes() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    removed_options = {
        "--incoming_batch_size",
        "--disable_dynamic_batching",
        "--alignment_mode",
        "--discard_frame_images_after_fusion",
        "--image_preprocess_cache",
        "--enable_image_preprocess_prefetch",
        "--pose_opt_cooldown_frames",
        "--pose_opt_max_tail_frames",
        "--pose_opt_window_frames",
    }
    assert option_strings.isdisjoint(removed_options)
    assert not hasattr(ReconstructionConfig(), "dynamic_batching")
    assert not hasattr(ReconstructionConfig(), "alignment_mode")
    assert ReconstructionConfig().pose_graph_mode == "final"


def test_public_ab05_config_uses_supported_arguments() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "incremental_ablation_template.yaml"
    args, info = parse_run_args(["--config", str(config_path), "--case", "ab05_all_on"])
    assert info["case"] == "ab05_all_on"
    assert args.max_batch_size == 5
    assert args.enable_register_ref_pruning_retry
    assert args.enable_pose_graph_optimization


def test_viewer_loads_config_and_preserves_viewer_arguments() -> None:
    from on_the_fly3r.viewer import build_parser as build_viewer_parser

    config_path = Path(__file__).resolve().parents[1] / "configs" / "incremental_ablation_template.yaml"
    args, info = parse_run_args(
        [
            "--config",
            str(config_path),
            "--image_dir",
            "/tmp/unused",
            "--output_dir",
            "/tmp/viewer-output",
            "--port",
            "9090",
        ],
        parser=build_viewer_parser(),
    )
    assert info["case"] == "ab05_all_on"
    assert args.model == "Pi3x"
    assert args.inference_device == "cuda:3"
    assert args.retrieval_device == "cuda:3"
    assert args.enable_register_ref_pruning_retry
    assert args.enable_pose_graph_optimization
    assert args.port == 9090


def test_local_spacing_sampler_is_callable_as_static_helper() -> None:
    points = np.zeros((2, 2, 3), dtype=np.float32)
    points[0, 1, 0] = 1.0
    points[1, 0, 1] = 1.0
    points[1, 1] = (1.0, 1.0, 0.0)
    confidence = np.ones((2, 2), dtype=np.float32)
    samples = IncrementalReconstructor._collect_local_spacing_samples(
        world_points=points,
        world_points_conf=confidence,
        conf_threshold=0.5,
        max_samples=16,
    )
    assert samples.shape == (4,)
    assert np.allclose(samples, 1.0)


def test_robust_sim3_recovers_known_transform() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(128, 3)).astype(np.float64)
    angle = np.deg2rad(20.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    scale = 1.25
    translation = np.array([0.4, -0.2, 0.7], dtype=np.float64)
    target = scale * (source @ rotation.T) + translation
    estimated_scale, estimated_rotation, estimated_translation = robust_weighted_estimate_sim3(
        source,
        target,
        np.ones((source.shape[0],), dtype=np.float64),
        use_scale=True,
    )
    assert estimated_scale == pytest.approx(scale, abs=1e-6)
    assert np.allclose(estimated_rotation, rotation, atol=1e-6)
    assert np.allclose(estimated_translation, translation, atol=1e-6)


def test_streaming_ply_and_numba_voxel_dedup(tmp_path: Path) -> None:
    points = [
        np.array(
            [[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            dtype=np.float32,
        )
    ]
    colors = [np.arange(12, dtype=np.uint8).reshape(4, 3)]
    raw_path = tmp_path / "raw.ply"
    dedup_path = tmp_path / "dedup.ply"
    assert _write_binary_ply_streaming(points, colors, str(raw_path)) == 4
    stats = _write_binary_ply_voxel_dedup_streaming(
        points,
        colors,
        str(dedup_path),
        voxel_size=0.5,
    )
    assert _ply_vertex_count(raw_path) == 4
    assert _ply_vertex_count(dedup_path) == 3
    assert stats["raw_vertex_count"] == 4
    assert stats["exported_vertex_count"] == 3


def test_gtsam_se3_pose_graph_reduces_cost() -> None:
    pytest.importorskip("gtsam")
    snapshot = PoseGraphSnapshot(
        frame_ids=["a", "b", "c"],
        cam2world=np.stack([_pose(0.0), _pose(1.2), _pose(2.4)], axis=0),
        edges=[
            PoseGraphEdge("a", "b", _pose(1.0), edge_type="local_continuity"),
            PoseGraphEdge("b", "c", _pose(1.0), edge_type="local_continuity"),
            PoseGraphEdge("a", "c", _pose(2.0), edge_type="loop", weight=3.0),
        ],
        snapshot_frame_count=3,
        trigger_reason="test",
    )
    result = optimize_pose_graph(snapshot, max_nfev=30)
    assert result.summary["success"]
    assert result.summary["backend"] == "gtsam"
    assert result.summary["model"] == "se3"
    assert result.summary["final_cost"] < result.summary["initial_cost"]


def test_cuda_frame_index_orders_cosine_similarity() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 3:
        pytest.skip("cuda:3 is required for the production retrieval index")
    index = FrameIndex(device_type="cuda:3")
    index.add("x", np.array([1.0, 0.0], dtype=np.float32))
    index.add("y", np.array([0.0, 1.0], dtype=np.float32))
    assert index.search(np.array([0.9, 0.1], dtype=np.float32), topk=2) == ["x", "y"]


def test_canonical_vfm_dispatch_preserves_prediction_fields(monkeypatch) -> None:
    from on_the_fly3r import vfm_adapter

    expected = {"cam2world": np.eye(4, dtype=np.float32)[None]}

    def fake_predictions(image_names, model_name, model, **kwargs):
        assert image_names == ["frame.jpg"]
        assert model_name == "Pi3x"
        assert model == "model"
        assert kwargs == {"return_profile": True}
        return expected

    monkeypatch.setattr(vfm_adapter, "run_predictions", fake_predictions)
    assert vfm_adapter.infer_vfm(
        ["frame.jpg"],
        "Pi3x",
        "model",
        return_profile=True,
    ) is expected

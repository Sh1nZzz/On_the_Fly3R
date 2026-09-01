from types import SimpleNamespace
import threading

import numpy as np

from on_the_fly3r.pose_graph import (
    PoseGraphEdge,
    PoseGraphMixin,
    PoseGraphResult,
)
from on_the_fly3r.state import ReconstructionState
from on_the_fly3r.types import FrameReconstruction


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = x
    return pose


def _frame(frame_id: str, x: float) -> FrameReconstruction:
    return FrameReconstruction(
        frame_id=frame_id,
        image_path=f"/tmp/{frame_id}.jpg",
        cam2world=_pose(x),
        intrinsic=np.eye(3, dtype=np.float32),
        world_points=None,
        world_points_conf=None,
        image=None,
    )


class _PoseGraphHarness(PoseGraphMixin):
    def __init__(self, *, mode: str = "online_and_final", interval: int = 2) -> None:
        self.config = SimpleNamespace(
            enable_pose_graph_optimization=True,
            pose_graph_mode=mode,
            pose_opt_interval_frames=interval,
            pose_opt_min_edges=1,
            pose_opt_min_loop_edges=1,
            pose_opt_max_nfev=5,
        )
        self.state = ReconstructionState()
        self.pose_graph_edges = []
        self.pose_graph_events = []
        self.runtime_stats = {}
        self._last_pose_graph_debug_payload = None
        self._online_pgo_accepted_frames_since_check = 0
        self._online_pgo_new_loop_edges_since_check = 0
        self._online_pgo_check_count = 0
        self._online_pgo_run_count = 0


def test_state_exports_optimized_pose_without_changing_map_pose() -> None:
    state = ReconstructionState()
    state.add_frame(_frame("a", 1.0))

    assert np.array_equal(state.get_map_pose("a"), _pose(1.0))
    assert np.array_equal(state.get_optimized_pose("a"), _pose(1.0))

    state.set_optimized_pose("a", _pose(2.0))
    exported = state.export_camera_poses(["/tmp/a.jpg"])

    assert np.array_equal(state.get_map_pose("a"), _pose(1.0))
    assert np.array_equal(exported["cam2world"], exported["cam2world_optimized"])
    assert np.array_equal(exported["cam2world_optimized"][0], _pose(2.0))
    assert np.array_equal(exported["cam2world_map"][0], _pose(1.0))


def test_pose_graph_merge_updates_only_optimized_pose() -> None:
    harness = _PoseGraphHarness()
    frame_a = _frame("a", 0.0)
    frame_b = _frame("b", 1.0)
    frame_b.metadata["alignment_compact_reference"] = {
        "points": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    }
    harness.state.add_frame(frame_a)
    harness.state.add_frame(frame_b)
    harness.state.global_points.append(
        np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32)
    )
    map_before = {
        frame_id: harness.state.get_map_pose(frame_id).copy()
        for frame_id in ("a", "b")
    }
    points_before = harness.state.global_points[0].copy()
    compact_before = frame_b.metadata["alignment_compact_reference"]["points"].copy()
    result = PoseGraphResult(
        frame_ids=["a", "b"],
        cam2world_initial=np.stack([_pose(0.0), _pose(1.0)]),
        cam2world_refined=np.stack([_pose(0.0), _pose(1.5)]),
        snapshot_frame_count=2,
        summary={
            "success": True,
            "initial_cost": 2.0,
            "final_cost": 1.0,
            "optimization_time_sec": 0.01,
        },
    )

    event = harness._merge_pose_graph_result(result)

    assert event["event"] == "pose_graph_merged"
    assert np.array_equal(harness.state.get_map_pose("a"), map_before["a"])
    assert np.array_equal(harness.state.get_map_pose("b"), map_before["b"])
    assert np.array_equal(harness.state.get_optimized_pose("b"), _pose(1.5))
    assert np.array_equal(harness.state.global_points[0], points_before)
    assert np.array_equal(
        frame_b.metadata["alignment_compact_reference"]["points"], compact_before
    )


def test_online_pgo_checks_only_at_interval_and_requires_new_loop(monkeypatch) -> None:
    harness = _PoseGraphHarness(mode="online", interval=2)
    harness.state.add_frame(_frame("a", 0.0))
    harness.state.add_frame(_frame("b", 1.0))
    harness.pose_graph_edges.append(
        PoseGraphEdge("a", "b", _pose(1.0), edge_type="loop")
    )
    harness._online_pgo_new_loop_edges_since_check = 1

    def fake_optimize(snapshot, *, max_nfev):
        del max_nfev
        return PoseGraphResult(
            frame_ids=list(snapshot.frame_ids),
            cam2world_initial=snapshot.cam2world.copy(),
            cam2world_refined=np.stack([_pose(0.0), _pose(1.25)]),
            snapshot_frame_count=snapshot.snapshot_frame_count,
            summary={
                "success": True,
                "initial_cost": 1.0,
                "final_cost": 0.5,
                "optimization_time_sec": 0.01,
            },
        )

    monkeypatch.setattr("on_the_fly3r.pose_graph.optimize_pose_graph", fake_optimize)

    assert harness.maybe_run_online_pose_graph_optimization(1) == []
    events = harness.maybe_run_online_pose_graph_optimization(1)

    assert len(events) == 1
    assert events[0]["event"] == "pose_graph_online_scheduled"
    assert events[0]["phase"] == "online"
    assert harness._online_pgo_run_count == 0
    assert np.array_equal(harness.state.get_map_pose("b"), _pose(1.0))
    assert np.array_equal(harness.state.get_optimized_pose("b"), _pose(1.0))

    assert harness.maybe_run_online_pose_graph_optimization(1) == []
    events = harness.maybe_run_online_pose_graph_optimization(1)
    assert len(events) == 1
    assert events[0]["event"] == "pose_graph_online_skipped"
    assert events[0]["reason"] == "no_new_loop_edges"

    events = harness.finalize_pose_graph_optimization()
    assert len(events) == 1
    assert events[0]["event"] == "pose_graph_merged"
    assert harness._online_pgo_run_count == 1
    assert np.array_equal(harness.state.get_map_pose("b"), _pose(1.0))
    assert np.array_equal(harness.state.get_optimized_pose("b"), _pose(1.25))


def test_online_pgo_submission_does_not_wait_for_worker(monkeypatch) -> None:
    harness = _PoseGraphHarness(mode="online", interval=1)
    harness.state.add_frame(_frame("a", 0.0))
    harness.state.add_frame(_frame("b", 1.0))
    harness.pose_graph_edges.append(
        PoseGraphEdge("a", "b", _pose(1.0), edge_type="loop")
    )
    harness._online_pgo_new_loop_edges_since_check = 1
    started = threading.Event()
    release = threading.Event()

    def fake_optimize(snapshot, *, max_nfev):
        del max_nfev
        started.set()
        assert release.wait(timeout=2.0)
        return PoseGraphResult(
            frame_ids=list(snapshot.frame_ids),
            cam2world_initial=snapshot.cam2world.copy(),
            cam2world_refined=snapshot.cam2world.copy(),
            snapshot_frame_count=snapshot.snapshot_frame_count,
            summary={
                "success": True,
                "initial_cost": 1.0,
                "final_cost": 0.5,
                "optimization_time_sec": 0.01,
            },
        )

    monkeypatch.setattr("on_the_fly3r.pose_graph.optimize_pose_graph", fake_optimize)

    events = harness.maybe_run_online_pose_graph_optimization(1)
    assert events[0]["event"] == "pose_graph_online_scheduled"
    assert started.wait(timeout=1.0)
    assert not release.is_set()

    release.set()
    harness.finalize_pose_graph_optimization()


def test_queued_online_pgo_uses_previous_result_as_next_initial_pose(monkeypatch) -> None:
    harness = _PoseGraphHarness(mode="online", interval=1)
    harness.state.add_frame(_frame("a", 0.0))
    harness.state.add_frame(_frame("b", 1.0))
    harness.pose_graph_edges.append(
        PoseGraphEdge("a", "b", _pose(1.0), edge_type="loop")
    )
    seen_initials = []

    def fake_optimize(snapshot, *, max_nfev):
        del max_nfev
        initial = snapshot.cam2world.copy()
        seen_initials.append(initial)
        refined = initial.copy()
        refined[-1, 0, 3] += 0.25
        return PoseGraphResult(
            frame_ids=list(snapshot.frame_ids),
            cam2world_initial=initial,
            cam2world_refined=refined,
            snapshot_frame_count=snapshot.snapshot_frame_count,
            summary={
                "success": True,
                "initial_cost": 1.0,
                "final_cost": 0.5,
                "optimization_time_sec": 0.01,
            },
        )

    monkeypatch.setattr("on_the_fly3r.pose_graph.optimize_pose_graph", fake_optimize)

    harness._online_pgo_new_loop_edges_since_check = 1
    harness.maybe_run_online_pose_graph_optimization(1)

    harness.state.add_frame(_frame("c", 2.0))
    harness.pose_graph_edges.append(
        PoseGraphEdge("b", "c", _pose(1.0), edge_type="loop")
    )
    harness._online_pgo_new_loop_edges_since_check = 1
    harness.maybe_run_online_pose_graph_optimization(1)
    harness.finalize_pose_graph_optimization()

    assert len(seen_initials) == 2
    assert seen_initials[0][1, 0, 3] == 1.0
    assert seen_initials[1][1, 0, 3] == 1.25

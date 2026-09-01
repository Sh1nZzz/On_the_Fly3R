from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import sys
from pathlib import Path
import threading
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import ReconstructionConfig
from .utils import _normalize_feature_vector, _resolve_runtime_device


@dataclass
class QueryContext:
    frame_id: str
    image_path: str
    feature: np.ndarray
    scored_neighbors: List[tuple[str, float]]


@dataclass
class BatchPlan:
    image_paths: List[str]
    frame_ids: List[str]
    query_features: Dict[str, np.ndarray]
    scored_neighbors_per_frame: Dict[str, List[tuple[str, float]]]
    ranked_neighbors: List[tuple[str, float]]
    selected_neighbors: List[str]
    retrieval_stats: Dict[str, float]
    formation_reason: str
    formation_diagnostics: List[Dict[str, object]] = field(default_factory=list)
    blocked_frame_id: Optional[str] = None


class FrameIndex:
    """CUDA-only brute-force cosine index for reconstructed frames."""

    def __init__(self, device_type: str = "cuda") -> None:
        self.frame_ids: List[str] = []
        self.features: List[np.ndarray] = []
        self._feature_bank_torch: Optional[torch.Tensor] = None
        self._bank_dirty = False
        self.device_type = _resolve_runtime_device(device_type)
        if not self.device_type.lower().startswith("cuda"):
            raise ValueError("FrameIndex requires a CUDA retrieval device.")
        if not torch.cuda.is_available():
            raise RuntimeError("FrameIndex requires CUDA for Torch GPU retrieval.")

    def add(self, frame_id: str, feature: np.ndarray) -> None:
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        self.frame_ids.append(frame_id)
        self.features.append(feature)
        self._bank_dirty = True

    def _ensure_feature_bank_torch(self) -> Optional[torch.Tensor]:
        if not self.features:
            return None
        if self._feature_bank_torch is None or self._bank_dirty:
            bank = np.stack(self.features, axis=0)
            self._feature_bank_torch = torch.from_numpy(bank).to(self.device_type)
            self._bank_dirty = False
        return self._feature_bank_torch

    def search(self, query_feature: np.ndarray, topk: int) -> List[str]:
        return [frame_id for frame_id, _ in self.search_with_scores(query_feature, topk)]

    def search_with_scores(self, query_feature: np.ndarray, topk: int) -> List[tuple[str, float]]:
        if not self.features:
            return []
        query = _normalize_feature_vector(query_feature)
        topk = min(int(topk), len(self.frame_ids))
        if topk <= 0:
            return []
        bank_torch = self._ensure_feature_bank_torch()
        if bank_torch is None:
            return []
        query_torch = torch.from_numpy(query).to(self.device_type)
        scores_torch = bank_torch @ query_torch
        values, indices = torch.topk(scores_torch, k=topk, largest=True, sorted=True)
        indices_np = indices.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy()
        return [(self.frame_ids[int(i)], float(score)) for i, score in zip(indices_np, values_np)]

    def release_gpu_resources(self) -> None:
        self._feature_bank_torch = None


class SupSceneRetrievalEncoder:
    """
    DINOv2 + SCPP overlap-aware retrieval wrapper from SupScene.
    """

    def __init__(self, config: ReconstructionConfig) -> None:
        self.config = config
        self.device = _resolve_runtime_device(config.retrieval_device)
        self.image_size = int(config.supscene_image_size)

        repo_root = Path(__file__).resolve().parents[1] / "SupScene"
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from hubconf import dinov2_scpp_supscene_1536
        from torchvision.transforms import v2 as T2

        self.weights_path = self._resolve_weights_path(config.supscene_weights)
        self.model = dinov2_scpp_supscene_1536(
            pretrained=True,
            weights=self.weights_path,
            map_location=self.device,
        ).eval()
        self.transform = T2.Compose(
            [
                T2.ToImage(),
                T2.Resize(
                    size=(self.image_size, self.image_size),
                    interpolation=T2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T2.ToDtype(torch.float32, scale=True),
                T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _resolve_weights_path(weights_path: Optional[str]) -> str:
        candidate_paths = []
        if weights_path:
            candidate_paths.append(Path(weights_path))
        candidate_paths.append(
            Path(__file__).resolve().parents[1] / "SupScene" / "weights" / "dinov2_scpp_supscene_1536.pth"
        )
        for path in candidate_paths:
            if path.exists():
                return str(path)
        searched = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(
            "SupScene weights not found. Please provide --supscene_weights, or place the released weight at "
            f"'SupScene/weights/dinov2_scpp_supscene_1536.pth'. Searched: {searched}"
        )

    def encode_path(self, image_path: str) -> np.ndarray:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            descriptor = self.model(tensor)
        return descriptor.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    def release_gpu_resources(self) -> None:
        """Drop the resident SupScene model after retrieval is complete."""
        self.model = None

def build_retrieval_encoder(config: ReconstructionConfig) -> SupSceneRetrievalEncoder:
    return SupSceneRetrievalEncoder(config)


class RetrievalManager:
    """Own the SupScene encoder, CUDA index, feature cache, and prefetch worker."""

    def __init__(
        self,
        config: ReconstructionConfig,
        feature_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        self.config = config
        self.index = FrameIndex(config.retrieval_device)
        self.encoder = build_retrieval_encoder(config)
        self.feature_fn = feature_fn or self.encoder.encode_path
        self.feature_cache: Dict[str, np.ndarray] = {}
        self.feature_futures: Dict[str, Future[np.ndarray]] = {}
        self.feature_lock = threading.Lock()
        self.prefetch_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrieval-prefetch")
            if config.enable_retrieval_prefetch
            else None
        )

    def encode_uncached(self, image_path: str) -> np.ndarray:
        with self.feature_lock:
            return _normalize_feature_vector(self.feature_fn(image_path))

    def get_feature(self, image_path: str) -> np.ndarray:
        cached = self.feature_cache.get(image_path)
        if cached is None:
            future = self.feature_futures.pop(image_path, None)
            cached = future.result() if future is not None else self.encode_uncached(image_path)
            self.feature_cache[image_path] = cached
        return cached

    def prefetch(self, image_paths: Sequence[str]) -> None:
        if self.prefetch_executor is None:
            return
        for image_path in image_paths[: max(0, int(self.config.retrieval_prefetch_count))]:
            if image_path in self.feature_cache or image_path in self.feature_futures:
                continue
            self.feature_futures[image_path] = self.prefetch_executor.submit(
                self.encode_uncached,
                image_path,
            )

    def shutdown(self, *, wait: bool = True) -> None:
        if self.prefetch_executor is not None:
            self.prefetch_executor.shutdown(wait=wait, cancel_futures=True)
            self.prefetch_executor = None
        self.feature_futures.clear()
        self.feature_cache.clear()
        self.index.release_gpu_resources()
        self.encoder.release_gpu_resources()


class RetrievalPlanningMixin:
    def _get_cached_retrieval_feature(self, image_path: str) -> np.ndarray:
        return self.retrieval_manager.get_feature(image_path)

    def prefetch_retrieval_features(self, image_paths: Sequence[str]) -> None:
        self.retrieval_manager.prefetch(image_paths)

    def _prepare_query_context(self, image_path: str) -> QueryContext:
        frame_id = Path(image_path).stem
        feature = self._get_cached_retrieval_feature(image_path)
        scored_neighbors = self.index.search_with_scores(feature, self.config.retrieval_topk)
        return QueryContext(
            frame_id=frame_id,
            image_path=image_path,
            feature=feature,
            scored_neighbors=scored_neighbors,
        )

    def _aggregate_scored_neighbors(
        self,
        scored_neighbors_per_frame: Dict[str, List[tuple[str, float]]],
    ) -> tuple[List[tuple[str, float]], List[str]]:
        aggregate_stats: Dict[str, Dict[str, float]] = {}
        for scored_neighbors in scored_neighbors_per_frame.values():
            for rank, (neighbor_id, score) in enumerate(scored_neighbors):
                stats = aggregate_stats.setdefault(
                    neighbor_id,
                    {
                        "support_count": 0.0,
                        "score_sum": 0.0,
                        "score_max": float("-inf"),
                        "rank_score_sum": 0.0,
                    },
                )
                stats["support_count"] += 1.0
                stats["score_sum"] += float(score)
                stats["score_max"] = max(float(stats["score_max"]), float(score))
                stats["rank_score_sum"] += 1.0 / float(rank + 1)

        ranked_items = sorted(
            aggregate_stats.items(),
            key=lambda item: (
                -item[1]["support_count"],
                -item[1]["score_sum"],
                -item[1]["score_max"],
                -item[1]["rank_score_sum"],
                item[0],
            ),
        )
        ranked_neighbors = [
            (
                neighbor_id,
                float(
                    stats["support_count"]
                    + 0.01 * stats["score_sum"]
                    + 0.001 * stats["score_max"]
                    + 0.0001 * stats["rank_score_sum"]
                ),
            )
            for neighbor_id, stats in ranked_items
        ]
        selected_neighbors = [neighbor_id for neighbor_id, _ in ranked_neighbors[: self.config.retrieval_topk]]
        return ranked_neighbors, selected_neighbors

    @staticmethod
    def _update_reference_counter(
        counter: Dict[str, Dict[str, float]],
        scored_neighbors: Sequence[tuple[str, float]],
    ) -> None:
        for neighbor_id, score in scored_neighbors:
            stats = counter.setdefault(
                neighbor_id,
                {"support_count": 0.0, "score_sum": 0.0, "score_max": float("-inf")},
            )
            stats["support_count"] += 1.0
            stats["score_sum"] += float(score)
            stats["score_max"] = max(float(stats["score_max"]), float(score))

    def _get_dominant_refs(self, counter: Dict[str, Dict[str, float]]) -> List[str]:
        if not counter:
            return []
        ranked = sorted(
            counter.items(),
            key=lambda item: (
                -item[1]["support_count"],
                -item[1]["score_sum"],
                -item[1]["score_max"],
                item[0],
            ),
        )
        limit = max(1, int(self.config.dynamic_batch_dominant_refs))
        return [frame_id for frame_id, _ in ranked[:limit]]

    def _is_query_compatible_with_batch(
        self,
        candidate_context: QueryContext,
        batch_contexts: Sequence[QueryContext],
        batch_ref_counter: Dict[str, Dict[str, float]],
    ) -> tuple[bool, Dict[str, object]]:
        dominant_refs = self._get_dominant_refs(batch_ref_counter)
        candidate_refs = [neighbor_id for neighbor_id, _ in candidate_context.scored_neighbors]
        dominant_ref_set = set(dominant_refs)
        candidate_ref_set = set(candidate_refs)
        overlap_count = len(dominant_ref_set & candidate_ref_set)
        overlap_ratio = float(overlap_count) / float(max(1, len(dominant_refs)))

        if batch_contexts:
            sims = [float(np.dot(context.feature, candidate_context.feature)) for context in batch_contexts]
            max_query_similarity = max(sims)
            mean_query_similarity = float(np.mean(sims))
        else:
            max_query_similarity = 1.0
            mean_query_similarity = 1.0

        compatible = (
            overlap_count >= int(self.config.dynamic_batch_ref_overlap_min)
            and (
                max_query_similarity >= float(self.config.dynamic_batch_query_similarity_min)
                or overlap_ratio >= float(self.config.dynamic_batch_ref_overlap_ratio_min)
            )
        )

        diagnostics = {
            "candidate_frame_id": candidate_context.frame_id,
            "compatible": compatible,
            "ref_overlap_count": int(overlap_count),
            "ref_overlap_ratio": overlap_ratio,
            "max_query_similarity": max_query_similarity,
            "mean_query_similarity": mean_query_similarity,
            "dominant_refs": dominant_refs,
            "candidate_refs": candidate_refs[: self.config.dynamic_batch_dominant_refs],
        }
        return compatible, diagnostics

    def _form_next_dynamic_batch(
        self,
        remaining_paths: Sequence[str],
        start_idx: int,
    ) -> tuple[BatchPlan, int]:
        cursor = int(start_idx)
        retrieval_time_sec = 0.0
        batch_contexts: List[QueryContext] = []
        batch_ref_counter: Dict[str, Dict[str, float]] = {}
        formation_diagnostics: List[Dict[str, object]] = []

        seed_context, stats = self._measure_stage(
            lambda path=remaining_paths[cursor]: self._prepare_query_context(path),
            device_type=self.config.retrieval_device,
        )
        retrieval_time_sec += stats["time_sec"]
        batch_contexts.append(seed_context)
        self._update_reference_counter(batch_ref_counter, seed_context.scored_neighbors)
        cursor += 1

        formation_reason = "end_of_sequence"
        while cursor < len(remaining_paths) and len(batch_contexts) < self.config.max_batch_size:
            candidate_context, stats = self._measure_stage(
                lambda path=remaining_paths[cursor]: self._prepare_query_context(path),
                device_type=self.config.retrieval_device,
            )
            retrieval_time_sec += stats["time_sec"]

            compatible, diagnostics = self._is_query_compatible_with_batch(
                candidate_context=candidate_context,
                batch_contexts=batch_contexts,
                batch_ref_counter=batch_ref_counter,
            )
            formation_diagnostics.append(diagnostics)
            if not compatible:
                formation_reason = "next_frame_incompatible"
                break

            batch_contexts.append(candidate_context)
            self._update_reference_counter(batch_ref_counter, candidate_context.scored_neighbors)
            cursor += 1

        if len(batch_contexts) >= self.config.max_batch_size:
            formation_reason = "reached_max_batch_size"
        elif cursor >= len(remaining_paths):
            formation_reason = "end_of_sequence"

        scored_neighbors_per_frame = {
            context.frame_id: context.scored_neighbors for context in batch_contexts
        }
        ranked_neighbors, selected_neighbors = self._aggregate_scored_neighbors(scored_neighbors_per_frame)
        batch_plan = BatchPlan(
            image_paths=[context.image_path for context in batch_contexts],
            frame_ids=[context.frame_id for context in batch_contexts],
            query_features={context.frame_id: context.feature for context in batch_contexts},
            scored_neighbors_per_frame=scored_neighbors_per_frame,
            ranked_neighbors=ranked_neighbors,
            selected_neighbors=selected_neighbors,
            retrieval_stats={"time_sec": retrieval_time_sec},
            formation_reason=formation_reason,
            formation_diagnostics=formation_diagnostics,
            blocked_frame_id=formation_diagnostics[-1]["candidate_frame_id"] if formation_reason == "next_frame_incompatible" and formation_diagnostics else None,
        )
        return batch_plan, cursor

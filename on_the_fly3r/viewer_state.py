from dataclasses import dataclass
from typing import Iterable


def playback_slider_state(latest_batch_idx: int) -> tuple[int, bool]:
    """Return a valid slider maximum and whether playback navigation is disabled."""
    latest_batch_idx = max(0, int(latest_batch_idx))
    return max(1, latest_batch_idx), latest_batch_idx <= 0


def cumulative_compute_time_sec(
    bootstrap_time_sec: float,
    batch_times: Iterable[tuple[int, float]],
    target_batch_idx: int,
    *,
    finalization_time_sec: float = 0.0,
    include_finalization: bool = False,
) -> float:
    """Compute reconstruction time accumulated through a selected batch."""
    total = max(0.0, float(bootstrap_time_sec))
    target_batch_idx = max(0, int(target_batch_idx))
    for batch_idx, duration_sec in batch_times:
        if int(batch_idx) > target_batch_idx:
            break
        total += max(0.0, float(duration_sec))
    if include_finalization:
        total += max(0.0, float(finalization_time_sec))
    return total


@dataclass
class PlaybackState:
    """Pure playback state shared by the Viser GUI and its timer loop."""

    is_playing: bool = False
    follow_live: bool = True
    loop_enabled: bool = False

    @staticmethod
    def _clamp(batch_idx: int, latest_batch_idx: int) -> int:
        return int(max(0, min(int(batch_idx), max(0, int(latest_batch_idx)))))

    def reset(self, *, loop_enabled: bool = False) -> None:
        self.is_playing = False
        self.follow_live = True
        self.loop_enabled = bool(loop_enabled)

    def on_latest_changed(self, current_batch_idx: int, latest_batch_idx: int) -> int:
        if self.follow_live:
            return max(0, int(latest_batch_idx))
        return self._clamp(current_batch_idx, latest_batch_idx)

    def select(self, batch_idx: int, latest_batch_idx: int) -> int:
        self.is_playing = False
        self.follow_live = False
        return self._clamp(batch_idx, latest_batch_idx)

    def go_live(self, latest_batch_idx: int) -> int:
        self.is_playing = False
        self.follow_live = True
        return max(0, int(latest_batch_idx))

    def start(self, current_batch_idx: int, latest_batch_idx: int) -> int:
        latest_batch_idx = max(0, int(latest_batch_idx))
        if latest_batch_idx <= 0:
            self.is_playing = False
            return 0
        self.follow_live = False
        current_batch_idx = self._clamp(current_batch_idx, latest_batch_idx)
        if current_batch_idx >= latest_batch_idx:
            current_batch_idx = 0
        self.is_playing = True
        return current_batch_idx

    def advance(self, current_batch_idx: int, latest_batch_idx: int) -> int:
        latest_batch_idx = max(0, int(latest_batch_idx))
        current_batch_idx = self._clamp(current_batch_idx, latest_batch_idx)
        if not self.is_playing or latest_batch_idx <= 0:
            self.is_playing = False
            return current_batch_idx
        if current_batch_idx < latest_batch_idx:
            next_batch_idx = current_batch_idx + 1
            if next_batch_idx >= latest_batch_idx and not self.loop_enabled:
                self.is_playing = False
            return next_batch_idx
        if self.loop_enabled:
            return 0
        self.is_playing = False
        return latest_batch_idx

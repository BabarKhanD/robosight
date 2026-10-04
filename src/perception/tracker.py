from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from ..utils.logger import get_logger
from .detector import Detection

log = get_logger("tracker")

DIRECTIONS = ["right", "down_right", "down", "down_left", "left", "up_left", "up", "up_right"]


class MotionState(BaseModel):
    velocity_px_s: Tuple[float, float]
    speed_px_s: float
    direction: str
    is_moving: bool
    depth_trend: str
    nearness_rate_per_s: float


class _DetectionBatch:
    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx: Any) -> "_DetectionBatch":
        return _DetectionBatch(self.xywh[idx], self.conf[idx], self.cls[idx])

    @property
    def xyxy(self) -> np.ndarray:
        x, y, w, h = self.xywh.T
        return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


class ByteTrackTracker:
    """ByteTrack multi-object tracker (Ultralytics implementation) driven by external detections."""

    def __init__(self, cfg: Dict[str, Any], frame_rate: float = 30.0) -> None:
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as exc:
            raise ImportError("ultralytics (and lapx) are required for ByteTrack") from exc
        self._cls = BYTETracker
        self._args = SimpleNamespace(
            track_high_thresh=cfg.get("track_high_thresh", 0.25),
            track_low_thresh=cfg.get("track_low_thresh", 0.1),
            new_track_thresh=cfg.get("new_track_thresh", 0.25),
            track_buffer=cfg.get("track_buffer", 30),
            match_thresh=cfg.get("match_thresh", 0.8),
            fuse_score=cfg.get("fuse_score", True),
        )
        self._frame_rate = int(round(frame_rate))
        self.reset()

    def reset(self) -> None:
        try:
            self._tracker = self._cls(self._args, frame_rate=self._frame_rate)
        except TypeError:
            self._tracker = self._cls(self._args)

    def update(self, detections: List[Detection], frame_bgr: Optional[np.ndarray] = None) -> None:
        for det in detections:
            det.track_id = -1
        if detections:
            xyxy = np.stack([d.bbox for d in detections]).astype(np.float32)
            xywh = np.stack(
                [
                    (xyxy[:, 0] + xyxy[:, 2]) / 2,
                    (xyxy[:, 1] + xyxy[:, 3]) / 2,
                    xyxy[:, 2] - xyxy[:, 0],
                    xyxy[:, 3] - xyxy[:, 1],
                ],
                axis=1,
            )
            conf = np.array([d.confidence for d in detections], dtype=np.float32)
            cls = np.array([d.class_id for d in detections], dtype=np.float32)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            xywh = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)
        tracks = self._tracker.update(_DetectionBatch(xywh, conf, cls), frame_bgr)
        tracks = np.asarray(tracks)
        if tracks.size == 0 or not detections:
            return
        self._assign(detections, xyxy, tracks)

    @staticmethod
    def _assign(detections: List[Detection], det_xyxy: np.ndarray, tracks: np.ndarray) -> None:
        if tracks.ndim == 2 and tracks.shape[1] >= 8:
            for row in tracks:
                idx = int(row[-1])
                if 0 <= idx < len(detections) and detections[idx].track_id < 0:
                    detections[idx].track_id = int(row[4])
            return
        ious = _iou_matrix(tracks[:, :4], det_xyxy)
        used = set()
        for t in np.argsort(-ious.max(axis=1)):
            order = np.argsort(-ious[t])
            for d in order:
                if ious[t, d] < 0.3:
                    break
                if int(d) not in used:
                    detections[int(d)].track_id = int(tracks[t, 4])
                    used.add(int(d))
                    break


class MotionEstimator:
    """Per-track image-plane velocity and relative-depth trend over a sliding window."""

    def __init__(
        self,
        window: int = 5,
        min_speed_px_s: float = 15.0,
        depth_trend_threshold: float = 0.05,
    ) -> None:
        self.window = max(2, window)
        self.min_speed = min_speed_px_s
        self.depth_thr = depth_trend_threshold
        self._history: Dict[int, Deque[Tuple[float, float, float, float]]] = {}

    def reset(self) -> None:
        self._history.clear()

    def update(
        self, track_id: int, centroid: Tuple[float, float], nearness: float, t: float
    ) -> Optional[MotionState]:
        hist = self._history.setdefault(track_id, deque(maxlen=self.window))
        hist.append((t, centroid[0], centroid[1], nearness))
        if len(hist) < 2:
            return None
        arr = np.asarray(hist, dtype=np.float64)
        ts = arr[:, 0] - arr[0, 0]
        if ts[-1] <= 1e-6:
            return None
        vx = float(np.polyfit(ts, arr[:, 1], 1)[0])
        vy = float(np.polyfit(ts, arr[:, 2], 1)[0])
        rate = float(np.polyfit(ts, arr[:, 3], 1)[0])
        speed = float(np.hypot(vx, vy))
        moving = speed >= self.min_speed
        if moving:
            angle = np.degrees(np.arctan2(vy, vx))
            direction = DIRECTIONS[int(np.round(angle / 45.0)) % 8]
        else:
            direction = "stationary"
        if rate > self.depth_thr:
            trend = "approaching"
        elif rate < -self.depth_thr:
            trend = "receding"
        else:
            trend = "steady"
        return MotionState(
            velocity_px_s=(round(vx, 2), round(vy, 2)),
            speed_px_s=round(speed, 2),
            direction=direction,
            is_moving=moving,
            depth_trend=trend,
            nearness_rate_per_s=round(rate, 4),
        )

    def prune(self, active_ids: List[int]) -> None:
        keep = set(active_ids)
        for tid in list(self._history):
            if tid not in keep:
                del self._history[tid]

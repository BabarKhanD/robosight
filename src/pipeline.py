from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import resolve_device
from .perception.depth_estimator import DepthEstimator, DepthResult
from .perception.detector import Detection, YOLODetector
from .perception.tracker import ByteTrackTracker, MotionEstimator
from .spatial.depth_aligner import DepthAligner
from .spatial.scene_graph import SceneGraph, SceneGraphBuilder
from .utils.logger import get_logger
from .utils.visualizer import compose_side_by_side, depth_panel, draw_scene

log = get_logger("pipeline")


@dataclass
class FrameResult:
    frame_index: int
    graph: SceneGraph
    detections: List[Detection]
    depth: DepthResult
    timings_ms: Dict[str, float] = field(default_factory=dict)


class RoboSightPipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        tracking_enabled: Optional[bool] = None,
        frame_rate: float = 30.0,
    ) -> None:
        self.config = config
        device = resolve_device(config.get("runtime", {}).get("device", "auto"))
        det_cfg, dep_cfg = config["detector"], config["depth"]
        trk_cfg, sp_cfg = config["tracking"], config["spatial"]
        self.device = device
        self.detector = YOLODetector(
            weights=det_cfg["weights"],
            device=device,
            conf=det_cfg["conf"],
            iou=det_cfg["iou"],
            imgsz=det_cfg["imgsz"],
            max_det=det_cfg["max_det"],
            class_names=det_cfg.get("class_names"),
        )
        self.depth_estimator = DepthEstimator(
            model_id=dep_cfg["model_id"],
            device=device,
            percentiles=tuple(dep_cfg["percentiles"]),
        )
        self.depth_interval = max(1, int(dep_cfg.get("interval", 1)))
        self.tracking_enabled = trk_cfg["enabled"] if tracking_enabled is None else tracking_enabled
        self.tracker = ByteTrackTracker(trk_cfg, frame_rate) if self.tracking_enabled else None
        self.motion = MotionEstimator(
            window=trk_cfg["motion_window"],
            min_speed_px_s=trk_cfg["min_speed_px_s"],
            depth_trend_threshold=trk_cfg["depth_trend_threshold"],
        )
        self.aligner = DepthAligner(
            near_threshold=sp_cfg["near_threshold"],
            far_threshold=sp_cfg["far_threshold"],
            erosion_px=sp_cfg["erosion_px"],
            min_mask_pixels=sp_cfg["min_mask_pixels"],
        )
        self.builder = SceneGraphBuilder(
            lr_margin_frac=sp_cfg["lr_margin_frac"],
            depth_margin=sp_cfg["depth_margin"],
            overlap_threshold=sp_cfg["overlap_threshold"],
        )
        self._last_depth: Optional[DepthResult] = None

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()
        self.motion.reset()
        self._last_depth = None

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int = 0, timestamp_s: float = 0.0) -> FrameResult:
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        detections = self.detector.detect(frame_bgr)
        timings["detect"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        stale = (
            self._last_depth is None
            or self._last_depth.nearness.shape != frame_bgr.shape[:2]
            or frame_index % self.depth_interval == 0
        )
        if stale:
            self._last_depth = self.depth_estimator.estimate(frame_bgr)
        depth = self._last_depth
        timings["depth"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if self.tracker is not None:
            self.tracker.update(detections, frame_bgr)
        else:
            for det in detections:
                det.track_id = det.index + 1
        timings["track"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        depths = self.aligner.align(depth.nearness, detections)
        motions = {}
        if self.tracker is not None:
            active = []
            for det, d in zip(detections, depths):
                if det.track_id < 0:
                    continue
                active.append(det.track_id)
                state = self.motion.update(det.track_id, det.centroid, d.median_nearness, timestamp_s)
                if state is not None:
                    motions[det.uid] = state
            self.motion.prune(active)
        graph = self.builder.build(frame_index, timestamp_s, frame_bgr.shape, detections, depths, motions)
        timings["spatial"] = (time.perf_counter() - t0) * 1000
        timings["total"] = sum(timings.values())
        return FrameResult(frame_index, graph, detections, depth, timings)

    def render(self, frame_bgr: np.ndarray, result: FrameResult) -> np.ndarray:
        out_cfg = self.config["output"]
        annotated = draw_scene(
            frame_bgr,
            result.detections,
            result.graph,
            show_masks=out_cfg["show_masks"],
            show_motion=out_cfg["show_motion"],
            arrow_seconds=out_cfg["arrow_seconds"],
        )
        if out_cfg["side_by_side_depth"]:
            return compose_side_by_side(annotated, depth_panel(result.depth.nearness))
        return annotated

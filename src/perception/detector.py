from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils.logger import get_logger

log = get_logger("detector")


@dataclass
class Detection:
    index: int
    class_id: int
    label: str
    confidence: float
    bbox: np.ndarray
    mask: Optional[np.ndarray] = None
    track_id: int = -1

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

    @property
    def uid(self) -> str:
        if self.track_id >= 0:
            return f"{self.label}_{self.track_id}"
        return f"{self.label}_u{self.index}"


def box_to_mask(bbox: Sequence[float], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = bbox
    mask = np.zeros((h, w), dtype=bool)
    xi1, yi1 = max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0)
    xi2, yi2 = min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
    if xi2 > xi1 and yi2 > yi1:
        mask[yi1:yi2, xi1:xi2] = True
    return mask


def polygons_to_masks(
    polygons: Sequence[np.ndarray], shape: Tuple[int, int]
) -> List[Optional[np.ndarray]]:
    h, w = shape
    masks: List[Optional[np.ndarray]] = []
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float32)
        if pts.ndim != 2 or len(pts) < 3:
            masks.append(None)
            continue
        canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
        masks.append(canvas.astype(bool))
    return masks


class YOLODetector:
    """YOLOv8-Seg wrapper producing Detection objects with full-resolution boolean masks."""

    def __init__(
        self,
        weights: str = "yolov8n-seg.pt",
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        max_det: int = 40,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required: pip install ultralytics") from exc
        log.info("Loading detector weights=%s device=%s", weights, device)
        try:
            self.model = YOLO(weights)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO weights '{weights}'. Check the path or network access."
            ) from exc
        self.names: Dict[int, str] = dict(self.model.names)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = self._ultralytics_device(device)
        self.class_ids = self._resolve_classes(class_names)

    @staticmethod
    def _ultralytics_device(device: str) -> str:
        if device.startswith("cuda"):
            return device.split(":")[1] if ":" in device else "0"
        return device

    def _resolve_classes(self, class_names: Optional[Sequence[str]]) -> Optional[List[int]]:
        if not class_names:
            return None
        lookup = {name.lower(): idx for idx, name in self.names.items()}
        ids = []
        for name in class_names:
            if name.lower() not in lookup:
                raise ValueError(f"Unknown class '{name}'. Valid: {sorted(lookup)}")
            ids.append(lookup[name.lower()])
        return ids

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or frame_bgr.ndim != 3:
            raise ValueError("detect() expects a BGR image array of shape (H, W, 3)")
        h, w = frame_bgr.shape[:2]
        result = self.model.predict(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.class_ids,
            max_det=self.max_det,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        masks: List[Optional[np.ndarray]]
        if result.masks is not None:
            masks = polygons_to_masks(result.masks.xy, (h, w))
        else:
            masks = [None] * len(xyxy)
        detections: List[Detection] = []
        for i, (box, c, k) in enumerate(zip(xyxy, confs, classes)):
            mask = masks[i] if i < len(masks) else None
            if mask is None or not mask.any():
                mask = box_to_mask(box, (h, w))
            detections.append(
                Detection(
                    index=i,
                    class_id=int(k),
                    label=self.names.get(int(k), str(k)),
                    confidence=float(c),
                    bbox=box,
                    mask=mask,
                )
            )
        return detections

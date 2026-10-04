from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from ..perception.detector import Detection

NEAR = "Near"
MID = "Mid-Range"
FAR = "Far"


@dataclass
class ObjectDepth:
    median_nearness: float
    p25: float
    p75: float
    bucket: str
    source: str
    pixels: int


class DepthAligner:
    """Aligns the per-frame relative depth map with instance masks to get one depth value per object."""

    def __init__(
        self,
        near_threshold: float = 0.66,
        far_threshold: float = 0.33,
        erosion_px: int = 3,
        min_mask_pixels: int = 20,
    ) -> None:
        if not 0.0 <= far_threshold < near_threshold <= 1.0:
            raise ValueError("Require 0 <= far_threshold < near_threshold <= 1")
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.erosion_px = erosion_px
        self.min_mask_pixels = min_mask_pixels

    def bucket(self, nearness: float) -> str:
        if nearness >= self.near_threshold:
            return NEAR
        if nearness >= self.far_threshold:
            return MID
        return FAR

    def _region(self, det: Detection, shape: Sequence[int]) -> tuple:
        h, w = shape[:2]
        mask = det.mask
        if mask is not None and mask.shape == (h, w):
            region = mask
            if self.erosion_px > 0:
                k = 2 * self.erosion_px + 1
                eroded = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
                if eroded.sum() >= self.min_mask_pixels:
                    region = eroded
            if region.sum() >= self.min_mask_pixels:
                return region, "mask"
        x1, y1, x2, y2 = det.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw, hh = (x2 - x1) / 4, (y2 - y1) / 4
        box = np.zeros((h, w), dtype=bool)
        xa, xb = max(int(cx - hw), 0), min(int(np.ceil(cx + hw)) + 1, w)
        ya, yb = max(int(cy - hh), 0), min(int(np.ceil(cy + hh)) + 1, h)
        if xb > xa and yb > ya:
            box[ya:yb, xa:xb] = True
        return box, "box_center"

    def align(self, nearness: np.ndarray, detections: Sequence[Detection]) -> List[ObjectDepth]:
        results: List[ObjectDepth] = []
        for det in detections:
            region, source = self._region(det, nearness.shape)
            values = nearness[region]
            if values.size == 0:
                results.append(ObjectDepth(0.0, 0.0, 0.0, FAR, "empty", 0))
                continue
            med = float(np.median(values))
            results.append(
                ObjectDepth(
                    median_nearness=med,
                    p25=float(np.percentile(values, 25)),
                    p75=float(np.percentile(values, 75)),
                    bucket=self.bucket(med),
                    source=source,
                    pixels=int(values.size),
                )
            )
        return results

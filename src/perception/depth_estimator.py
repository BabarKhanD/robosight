from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from ..utils.logger import get_logger

log = get_logger("depth")


@dataclass
class DepthResult:
    raw: np.ndarray
    nearness: np.ndarray


def normalize_depth(raw: np.ndarray, percentiles: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    lo, hi = np.percentile(raw, percentiles)
    if hi - lo < 1e-6:
        return np.zeros_like(raw, dtype=np.float32)
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


class DepthEstimator:
    """Depth Anything V2 wrapper.

    The model outputs relative inverse depth (larger = closer), affine-invariant and not metric.
    `nearness` is that output normalized per frame to [0, 1] via percentiles (1.0 = nearest).
    """

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cpu",
        percentiles: Tuple[float, float] = (1.0, 99.0),
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise ImportError("torch and transformers are required for depth estimation") from exc
        self._torch = torch
        self.device = device
        self.percentiles = tuple(percentiles)
        log.info("Loading depth model=%s device=%s", model_id, device)
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load depth model '{model_id}'. First run needs internet access "
                "to download from Hugging Face; afterwards it is cached."
            ) from exc

    def estimate(self, frame_bgr: np.ndarray) -> DepthResult:
        if frame_bgr is None or frame_bgr.ndim != 3:
            raise ValueError("estimate() expects a BGR image array of shape (H, W, 3)")
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        torch = self._torch
        with torch.inference_mode():
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            pred = self.model(**inputs).predicted_depth
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            )
        raw = pred.squeeze().float().cpu().numpy()
        return DepthResult(raw=raw, nearness=normalize_depth(raw, self.percentiles))

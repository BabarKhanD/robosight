from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ..perception.detector import Detection
from ..spatial.depth_aligner import FAR, MID, NEAR
from ..spatial.scene_graph import SceneGraph

BUCKET_COLORS = {NEAR: (60, 60, 255), MID: (0, 165, 255), FAR: (255, 140, 40)}


def colorize_depth(nearness: np.ndarray) -> np.ndarray:
    u8 = (np.clip(nearness, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def _put_label(img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    x = int(max(0, min(x, img.shape[1] - tw - 6)))
    y = max(y, th + base + 2)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - base), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_scene(
    frame_bgr: np.ndarray,
    detections: Sequence[Detection],
    graph: SceneGraph,
    show_masks: bool = True,
    show_motion: bool = True,
    mask_alpha: float = 0.35,
    arrow_seconds: float = 0.5,
) -> np.ndarray:
    out = frame_bgr.copy()
    if show_masks:
        layer = out.copy()
        for det, obj in zip(detections, graph.objects):
            if det.mask is not None and det.mask.shape == out.shape[:2]:
                layer[det.mask] = BUCKET_COLORS.get(obj.depth_bucket, (200, 200, 200))
        out = cv2.addWeighted(layer, mask_alpha, out, 1.0 - mask_alpha, 0)
    for obj in graph.objects:
        color = BUCKET_COLORS.get(obj.depth_bucket, (200, 200, 200))
        x1, y1, x2, y2 = (int(v) for v in obj.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _put_label(out, f"{obj.uid} | {obj.depth_bucket} {obj.relative_nearness:.2f}", (x1, y1), color)
        m = obj.motion
        if show_motion and m is not None and m.is_moving:
            cx, cy = int(obj.centroid[0]), int(obj.centroid[1])
            end = (int(cx + m.velocity_px_s[0] * arrow_seconds), int(cy + m.velocity_px_s[1] * arrow_seconds))
            cv2.arrowedLine(out, (cx, cy), end, (255, 255, 255), 2, tipLength=0.3)
            cv2.putText(out, m.depth_trend, (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def depth_panel(nearness: np.ndarray) -> np.ndarray:
    panel = colorize_depth(nearness)
    cv2.putText(panel, "relative depth (bright = near, not metric)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def compose_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / right.shape[0]
        right = cv2.resize(right, (int(right.shape[1] * scale), left.shape[0]))
    return np.hstack([left, right])


def draw_hud(img: np.ndarray, lines: Sequence[str], origin: Optional[Tuple[int, int]] = None) -> np.ndarray:
    x, y = origin or (8, img.shape[0] - 8 - 18 * (len(lines) - 1))
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, y + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img

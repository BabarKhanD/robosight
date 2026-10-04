from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel

from ..perception.detector import Detection
from ..perception.tracker import MotionState
from .depth_aligner import ObjectDepth

DEPTH_SEMANTICS = (
    "relative inverse depth from a monocular model, normalized per frame to [0,1] (1 = nearest). "
    "Not metric; not comparable across frames or images."
)
PREDICATES = ("left_of", "right_of", "closer_than", "behind", "overlay")


class SceneObject(BaseModel):
    uid: str
    label: str
    class_id: int
    confidence: float
    track_id: Optional[int]
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    area_px: int
    relative_nearness: float
    nearness_iqr: Tuple[float, float]
    depth_bucket: str
    depth_source: str
    motion: Optional[MotionState] = None


class Relation(BaseModel):
    subject: str
    predicate: str
    object: str


class SceneGraph(BaseModel):
    frame_index: int
    timestamp_s: float
    width: int
    height: int
    depth_semantics: str = DEPTH_SEMANTICS
    objects: List[SceneObject]
    relations: List[Relation]

    def to_json(self, indent: Optional[int] = 2) -> str:
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def overlap_ratio(a: Detection, b: Detection) -> float:
    """Intersection area divided by the smaller object's area (mask-based when masks exist)."""
    if a.mask is not None and b.mask is not None and a.mask.shape == b.mask.shape:
        inter = float(np.logical_and(a.mask, b.mask).sum())
        denom = float(min(a.mask.sum(), b.mask.sum()))
    else:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        denom = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return inter / denom if denom > 0 else 0.0


class SceneGraphBuilder:
    """Builds the 2.5D scene graph: objects with relative depth plus pairwise spatial predicates.

    Predicates (subject, predicate, object):
      left_of / right_of : image-plane centroid separation exceeds lr_margin_frac * mean box width
      closer_than        : subject's relative nearness exceeds object's by more than depth_margin
      behind             : masks overlap and subject is farther than object by more than depth_margin
      overlay            : masks overlap with no clear depth order (symmetric; emitted once per pair)
    """

    def __init__(
        self,
        lr_margin_frac: float = 0.25,
        depth_margin: float = 0.10,
        overlap_threshold: float = 0.30,
    ) -> None:
        self.lr_margin_frac = lr_margin_frac
        self.depth_margin = depth_margin
        self.overlap_threshold = overlap_threshold

    def build(
        self,
        frame_index: int,
        timestamp_s: float,
        image_shape: Sequence[int],
        detections: Sequence[Detection],
        depths: Sequence[ObjectDepth],
        motions: Optional[Dict[str, MotionState]] = None,
    ) -> SceneGraph:
        if len(detections) != len(depths):
            raise ValueError("detections and depths must have equal length")
        motions = motions or {}
        objects: List[SceneObject] = []
        for det, d in zip(detections, depths):
            x1, y1, x2, y2 = (float(v) for v in det.bbox)
            area = int(det.mask.sum()) if det.mask is not None else int((x2 - x1) * (y2 - y1))
            objects.append(
                SceneObject(
                    uid=det.uid,
                    label=det.label,
                    class_id=det.class_id,
                    confidence=round(det.confidence, 4),
                    track_id=det.track_id if det.track_id >= 0 else None,
                    bbox=(round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
                    centroid=tuple(round(v, 1) for v in det.centroid),
                    area_px=area,
                    relative_nearness=round(d.median_nearness, 4),
                    nearness_iqr=(round(d.p25, 4), round(d.p75, 4)),
                    depth_bucket=d.bucket,
                    depth_source=d.source,
                    motion=motions.get(det.uid),
                )
            )
        relations = self._relations(detections, objects)
        h, w = image_shape[:2]
        return SceneGraph(
            frame_index=frame_index,
            timestamp_s=round(timestamp_s, 4),
            width=int(w),
            height=int(h),
            objects=objects,
            relations=relations,
        )

    def _relations(
        self, detections: Sequence[Detection], objects: Sequence[SceneObject]
    ) -> List[Relation]:
        rels: List[Relation] = []
        n = len(objects)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = objects[i], objects[j]
                da, db = detections[i], detections[j]
                wa = a.bbox[2] - a.bbox[0]
                wb = b.bbox[2] - b.bbox[0]
                dx = b.centroid[0] - a.centroid[0]
                lr_thr = self.lr_margin_frac * (wa + wb) / 2.0
                if dx > lr_thr:
                    rels.append(Relation(subject=a.uid, predicate="left_of", object=b.uid))
                    rels.append(Relation(subject=b.uid, predicate="right_of", object=a.uid))
                elif dx < -lr_thr:
                    rels.append(Relation(subject=b.uid, predicate="left_of", object=a.uid))
                    rels.append(Relation(subject=a.uid, predicate="right_of", object=b.uid))
                dn = a.relative_nearness - b.relative_nearness
                a_nearer = dn > self.depth_margin
                b_nearer = dn < -self.depth_margin
                if a_nearer:
                    rels.append(Relation(subject=a.uid, predicate="closer_than", object=b.uid))
                elif b_nearer:
                    rels.append(Relation(subject=b.uid, predicate="closer_than", object=a.uid))
                if overlap_ratio(da, db) >= self.overlap_threshold:
                    if a_nearer:
                        rels.append(Relation(subject=b.uid, predicate="behind", object=a.uid))
                    elif b_nearer:
                        rels.append(Relation(subject=a.uid, predicate="behind", object=b.uid))
                    else:
                        rels.append(Relation(subject=a.uid, predicate="overlay", object=b.uid))
        return rels

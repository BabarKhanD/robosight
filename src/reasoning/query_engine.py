from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..spatial.scene_graph import SceneGraph, SceneObject

NOTE = "Depth is relative (not metric)."


@dataclass
class QueryResult:
    question: str
    intent: str
    answer: str
    objects: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_RELATION_PATTERNS: List[Tuple[str, str]] = [
    (r"(?:to the )?left of (.+)", "left_of"),
    (r"(?:to the )?right of (.+)", "right_of"),
    (r"(?:closer|nearer) (?:to the camera )?than (.+)", "closer_than"),
    (r"(?:farther|further) (?:from the camera )?than (.+)", "farther_than"),
    (r"in front of (.+)", "in_front_of"),
    (r"behind (.+)", "behind"),
    (r"(?:overlap(?:s|ping)?|overlay(?:s|ing)?) (?:with )?(.+)", "overlay"),
]


class QueryEngine:
    """Rules-based robot query layer over a single SceneGraph."""

    def __init__(self, graph: SceneGraph) -> None:
        self.graph = graph
        self._by_uid: Dict[str, SceneObject] = {o.uid: o for o in graph.objects}
        self._rels: Set[Tuple[str, str, str]] = {
            (r.subject, r.predicate, r.object) for r in graph.relations
        }
        self._labels: List[str] = sorted({o.label.lower() for o in graph.objects}, key=len, reverse=True)

    def resolve(self, ref: str) -> List[SceneObject]:
        text = re.sub(r"^(?:the|a|an)\s+", "", ref.lower().strip(" ?.!"))
        if not text:
            return []
        for uid, obj in self._by_uid.items():
            if uid.lower() == text.replace(" ", "_") or uid.lower() == text:
                return [obj]
        m = re.fullmatch(r"(.+?)[\s#_]*(\d+)", text)
        if m:
            uid = f"{m.group(1).strip()}_{m.group(2)}"
            for key, obj in self._by_uid.items():
                if key.lower() == uid:
                    return [obj]
        text = {"people": "person", "persons": "person"}.get(text, text)
        singular = text[:-1] if text.endswith("s") else text
        return [o for o in self.graph.objects if o.label.lower() in (text, singular)]

    def _label_in(self, text: str) -> Optional[str]:
        for label in self._labels:
            if re.search(rf"\b{re.escape(label)}s?\b", text):
                return label
        return None

    def _fmt(self, obj: SceneObject) -> str:
        return f"{obj.uid} ({obj.depth_bucket}, nearness {obj.relative_nearness:.2f})"

    def closest(self, label: Optional[str] = None) -> QueryResult:
        pool = self.resolve(label) if label else list(self.graph.objects)
        if not pool:
            return QueryResult("closest", "closest", "No matching objects in the scene.")
        best = max(pool, key=lambda o: o.relative_nearness)
        return QueryResult("closest", "closest", f"Closest to camera: {self._fmt(best)}. {NOTE}", [best.uid])

    def farthest(self, label: Optional[str] = None) -> QueryResult:
        pool = self.resolve(label) if label else list(self.graph.objects)
        if not pool:
            return QueryResult("farthest", "farthest", "No matching objects in the scene.")
        worst = min(pool, key=lambda o: o.relative_nearness)
        return QueryResult("farthest", "farthest", f"Farthest from camera: {self._fmt(worst)}. {NOTE}", [worst.uid])

    def count(self, label: str) -> QueryResult:
        found = self.resolve(label)
        word = label.strip()
        return QueryResult(
            "count", "count", f"{len(found)} '{word}' object(s) in the scene.", [o.uid for o in found]
        )

    def relational(self, predicate: str, ref: str) -> QueryResult:
        refs = self.resolve(ref)
        if not refs:
            return QueryResult(ref, predicate, f"Could not find '{ref}' in the scene.")
        ref_ids = {r.uid for r in refs}
        subjects: List[str] = []
        for s, p, o in self._rels:
            if s in ref_ids:
                if p == "overlay" and predicate == "overlay" and o not in ref_ids:
                    subjects.append(o)
                elif p == "closer_than" and predicate == "farther_than" and o not in ref_ids:
                    subjects.append(o)
                elif p == "behind" and predicate == "in_front_of" and o not in ref_ids:
                    subjects.append(o)
            if o in ref_ids and s not in ref_ids:
                if p == predicate or (p == "overlay" and predicate == "overlay"):
                    subjects.append(s)
        subjects = sorted(set(subjects))
        names = ", ".join(r.uid for r in refs)
        phrase = predicate.replace("_", " ")
        if not subjects:
            return QueryResult(ref, predicate, f"Nothing is {phrase} {names}.")
        return QueryResult(ref, predicate, f"{phrase} {names}: {', '.join(subjects)}.", subjects)

    def nearest_to(self, ref: str) -> QueryResult:
        refs = self.resolve(ref)
        if not refs:
            return QueryResult(ref, "nearest_to", f"Could not find '{ref}' in the scene.")
        ref_ids = {r.uid for r in refs}
        best: Optional[Tuple[float, SceneObject]] = None
        for o in self.graph.objects:
            if o.uid in ref_ids:
                continue
            d = min(
                float(np.hypot(o.centroid[0] - r.centroid[0], o.centroid[1] - r.centroid[1]))
                for r in refs
            )
            if best is None or d < best[0]:
                best = (d, o)
        if best is None:
            return QueryResult(ref, "nearest_to", "No other objects in the scene.")
        return QueryResult(
            ref, "nearest_to", f"Nearest in the image plane to {ref}: {best[1].uid} ({best[0]:.0f} px).", [best[1].uid]
        )

    def by_bucket(self, bucket: str) -> QueryResult:
        found = [o for o in self.graph.objects if o.depth_bucket == bucket]
        if not found:
            return QueryResult(bucket, "bucket", f"No {bucket} objects.")
        return QueryResult(bucket, "bucket", f"{bucket}: {', '.join(o.uid for o in found)}.", [o.uid for o in found])

    def by_motion(self, kind: str) -> QueryResult:
        found: List[SceneObject] = []
        for o in self.graph.objects:
            m = o.motion
            if m is None:
                continue
            if kind == "moving" and m.is_moving:
                found.append(o)
            elif kind == "stationary" and not m.is_moving:
                found.append(o)
            elif kind in ("approaching", "receding") and m.depth_trend == kind:
                found.append(o)
        if not found:
            return QueryResult(kind, kind, f"No {kind} objects (needs video with tracking).")
        text = ", ".join(
            f"{o.uid}" + (f" {o.motion.direction}" if kind == "moving" else "") for o in found
        )
        return QueryResult(kind, kind, f"{kind.capitalize()}: {text}.", [o.uid for o in found])

    def describe(self) -> QueryResult:
        objs = self.graph.objects
        if not objs:
            return QueryResult("describe", "describe", "The scene contains no detected objects.")
        counts: Dict[str, int] = {}
        for o in objs:
            counts[o.label] = counts.get(o.label, 0) + 1
        summary = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
        near = max(objs, key=lambda o: o.relative_nearness)
        far = min(objs, key=lambda o: o.relative_nearness)
        text = (
            f"{len(objs)} object(s): {summary}. Closest: {near.uid} ({near.depth_bucket}). "
            f"Farthest: {far.uid} ({far.depth_bucket}). {NOTE}"
        )
        return QueryResult("describe", "describe", text, [o.uid for o in objs])

    def ask(self, question: str) -> QueryResult:
        q = question.lower().strip().rstrip("?.! ")
        res = self._dispatch(q)
        res.question = question
        return res

    def _dispatch(self, q: str) -> QueryResult:
        m = re.search(r"\bhow many (.+)", q)
        if m:
            rest = re.sub(r"\bpeople\b", "person", m.group(1))
            label = self._label_in(rest) or re.split(r"\s+(?:are|is|in|do|can)\b", rest)[0].strip()
            return self.count(label)
        m = re.search(r"\b(?:closest|nearest) to (?!(?:the )?camera\b)(?:the )?(.+)", q)
        if m:
            return self.nearest_to(m.group(1))
        if re.search(r"\b(closest|nearest)\b", q):
            return self.closest(self._label_in(q))
        if re.search(r"\b(farthest|furthest)\b", q):
            return self.farthest(self._label_in(q))
        for pattern, predicate in _RELATION_PATTERNS:
            m = re.search(pattern, q)
            if m:
                return self.relational(predicate, m.group(1))
        for kind in ("approaching", "receding", "stationary", "moving"):
            if re.search(rf"\b{kind}\b", q):
                return self.by_motion(kind)
        if re.search(r"\bmid[- ]?range\b", q):
            return self.by_bucket("Mid-Range")
        if re.search(r"\bnear\b", q):
            return self.by_bucket("Near")
        if re.search(r"\bfar\b", q):
            return self.by_bucket("Far")
        if re.search(r"\b(describe|summary|summari[sz]e|what do you see|scene)\b", q):
            return self.describe()
        label = self._label_in(q)
        if label:
            found = self.resolve(label)
            return QueryResult(q, "find", f"{len(found)} '{label}': {', '.join(o.uid for o in found)}.", [o.uid for o in found])
        return QueryResult(q, "unsupported", "Unsupported question. Try: closest/farthest, left/right of X, behind X, how many X, moving, describe.")

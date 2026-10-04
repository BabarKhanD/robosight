import numpy as np

from src.perception.detector import Detection, box_to_mask
from src.perception.tracker import MotionState
from src.reasoning.query_engine import QueryEngine
from src.spatial.depth_aligner import FAR, MID, NEAR, DepthAligner
from src.spatial.scene_graph import SceneGraphBuilder, overlap_ratio

SHAPE = (100, 200)


def det(i, label, box, track_id=None):
    return Detection(
        index=i, class_id=i, label=label, confidence=0.9,
        bbox=np.array(box, dtype=np.float32), mask=box_to_mask(box, SHAPE),
        track_id=i + 1 if track_id is None else track_id,
    )


def nearness_map():
    n = np.zeros(SHAPE, dtype=np.float32)
    n[:, :60] = 0.9
    n[:, 60:140] = 0.5
    n[:, 140:] = 0.1
    return n


def test_buckets():
    a = DepthAligner(0.66, 0.33)
    assert a.bucket(0.9) == NEAR and a.bucket(0.5) == MID and a.bucket(0.1) == FAR


def test_aligner_median_with_masks():
    dets = [det(0, "a", [5, 10, 55, 90]), det(1, "b", [145, 10, 195, 90])]
    out = DepthAligner(erosion_px=2).align(nearness_map(), dets)
    assert out[0].bucket == NEAR and out[0].source == "mask"
    assert out[1].bucket == FAR
    assert abs(out[0].median_nearness - 0.9) < 1e-6


def test_aligner_box_fallback_when_mask_missing():
    d = det(0, "a", [5, 10, 55, 90])
    d.mask = None
    out = DepthAligner().align(nearness_map(), [d])
    assert out[0].source == "box_center" and out[0].bucket == NEAR


def test_aligner_rejects_bad_thresholds():
    try:
        DepthAligner(near_threshold=0.3, far_threshold=0.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def build_graph():
    dets = [
        det(0, "person", [5, 10, 55, 90]),
        det(1, "car", [145, 10, 195, 90]),
        det(2, "chair", [65, 20, 135, 80]),
    ]
    depths = DepthAligner(erosion_px=2).align(nearness_map(), dets)
    return SceneGraphBuilder().build(0, 0.0, (*SHAPE, 3), dets, depths)


def rel_set(graph):
    return {(r.subject, r.predicate, r.object) for r in graph.relations}


def test_left_right_and_depth_relations():
    g = build_graph()
    rels = rel_set(g)
    assert ("person_1", "left_of", "car_2") in rels
    assert ("car_2", "right_of", "person_1") in rels
    assert ("person_1", "closer_than", "car_2") in rels
    assert ("car_2", "closer_than", "person_1") not in rels
    assert g.objects[0].depth_bucket == NEAR and g.objects[1].depth_bucket == FAR


def test_behind_and_overlay():
    dets = [det(0, "box", [10, 10, 90, 90]), det(1, "wall", [30, 20, 120, 95])]
    near = np.full(SHAPE, 0.5, dtype=np.float32)
    near[:, :60] = 0.9
    depths = DepthAligner(erosion_px=0).align(near, dets)
    g = SceneGraphBuilder().build(0, 0.0, (*SHAPE, 3), dets, depths)
    assert ("wall_2", "behind", "box_1") in rel_set(g)
    flat = np.full(SHAPE, 0.5, dtype=np.float32)
    g2 = SceneGraphBuilder().build(0, 0.0, (*SHAPE, 3), dets, DepthAligner(erosion_px=0).align(flat, dets))
    preds = [r.predicate for r in g2.relations]
    assert preds.count("overlay") == 1


def test_overlap_ratio():
    a, b = det(0, "a", [0, 0, 50, 50]), det(1, "b", [25, 0, 75, 50])
    assert abs(overlap_ratio(a, b) - 0.5) < 1e-6
    c = det(2, "c", [80, 0, 90, 50])
    assert overlap_ratio(a, c) == 0.0


def test_graph_json_roundtrip_and_length_check():
    g = build_graph()
    assert '"relative_nearness"' in g.to_json()
    assert g.depth_semantics.startswith("relative")
    try:
        SceneGraphBuilder().build(0, 0.0, (*SHAPE, 3), [det(0, "a", [0, 0, 5, 5])], [])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_query_engine():
    q = QueryEngine(build_graph())
    assert q.ask("Which object is closest to the camera?").objects == ["person_1"]
    assert q.ask("What is the farthest object?").objects == ["car_2"]
    assert set(q.ask("What is to the left of the car?").objects) == {"person_1", "chair_3"}
    assert set(q.ask("what is right of person 1").objects) == {"car_2", "chair_3"}
    assert q.ask("How many people are there?").objects == ["person_1"]
    assert q.ask("how many cars").objects == ["car_2"]
    assert "car_2" in q.ask("which objects are farther than the person").objects
    assert q.ask("what is nearest to the car").objects
    assert q.ask("what is far").objects == ["car_2"]
    assert q.ask("describe the scene").intent == "describe"
    assert q.ask("left of unicorn").answer.startswith("Could not find")
    assert q.ask("blorp").intent == "unsupported"


def test_query_motion():
    dets = [det(0, "person", [5, 10, 55, 90])]
    depths = DepthAligner().align(nearness_map(), dets)
    m = MotionState(velocity_px_s=(80.0, 0.0), speed_px_s=80.0, direction="right",
                    is_moving=True, depth_trend="approaching", nearness_rate_per_s=0.2)
    g = SceneGraphBuilder().build(1, 0.1, (*SHAPE, 3), dets, depths, {"person_1": m})
    q = QueryEngine(g)
    assert q.ask("what is moving").objects == ["person_1"]
    assert q.ask("what is approaching").objects == ["person_1"]
    assert q.ask("what is receding").objects == []

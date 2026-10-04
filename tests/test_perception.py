import numpy as np
import pytest

from src.perception.depth_estimator import normalize_depth
from src.perception.detector import Detection, box_to_mask, polygons_to_masks
from src.perception.tracker import MotionEstimator


def test_detection_uid_and_centroid():
    d = Detection(0, 0, "person", 0.9, np.array([0, 0, 10, 20], dtype=np.float32), track_id=3)
    assert d.uid == "person_3" and d.centroid == (5.0, 10.0)
    d.track_id = -1
    assert d.uid == "person_u0"


def test_box_to_mask_clips_to_image():
    m = box_to_mask([-5, -5, 10, 10], (20, 20))
    assert m.sum() == 100 and m[0, 0] and not m[10, 10]
    assert box_to_mask([30, 30, 40, 40], (20, 20)).sum() == 0


def test_polygons_to_masks():
    square = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)
    masks = polygons_to_masks([square, np.zeros((0, 2))], (50, 50))
    assert masks[0] is not None and masks[0][20, 20] and not masks[0][40, 40]
    assert masks[1] is None


def test_normalize_depth_range_and_order():
    raw = np.linspace(0, 10, 10000, dtype=np.float32).reshape(100, 100)
    n = normalize_depth(raw)
    assert n.min() >= 0.0 and n.max() <= 1.0
    assert n[99, 99] > n[0, 0]
    assert normalize_depth(np.ones((4, 4))).max() == 0.0


def test_motion_estimator_moving_and_approaching():
    me = MotionEstimator(window=5, min_speed_px_s=15, depth_trend_threshold=0.05)
    state = None
    for k in range(5):
        t = k * 0.1
        state = me.update(1, (100 + 20 * k, 50), 0.3 + 0.05 * k, t)
    assert state is not None and state.is_moving
    assert state.direction == "right"
    assert abs(state.velocity_px_s[0] - 200.0) < 1.0
    assert state.depth_trend == "approaching"


def test_motion_estimator_stationary_and_prune():
    me = MotionEstimator()
    state = None
    for k in range(4):
        state = me.update(7, (10, 10), 0.5, k * 0.1)
    assert state.direction == "stationary" and state.depth_trend == "steady"
    assert me.update(8, (0, 0), 0.5, 0.0) is None
    me.prune([8])
    assert me.update(7, (10, 10), 0.5, 1.0) is None


def test_motion_estimator_ignores_zero_dt():
    me = MotionEstimator()
    me.update(1, (0, 0), 0.5, 0.0)
    assert me.update(1, (50, 0), 0.5, 0.0) is None


def test_bytetrack_keeps_identity_across_frames():
    pytest.importorskip("ultralytics")
    from src.perception.tracker import ByteTrackTracker

    cfg = {"track_high_thresh": 0.25, "track_low_thresh": 0.1, "new_track_thresh": 0.25,
           "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True}
    tr = ByteTrackTracker(cfg)
    ids = []
    for k in range(4):
        dets = [
            Detection(0, 0, "person", 0.9, np.array([10 + 2 * k, 10, 60 + 2 * k, 110], dtype=np.float32)),
            Detection(1, 2, "car", 0.8, np.array([200, 50, 320, 140], dtype=np.float32)),
        ]
        tr.update(dets)
        ids.append([d.track_id for d in dets])
    assert ids[-1][0] >= 0 and ids[-1][1] >= 0 and ids[-1][0] != ids[-1][1]
    assert all(row == ids[-1] for row in ids[1:])


def test_detector_and_depth_wrappers_need_models():
    pytest.importorskip("ultralytics")
    pytest.importorskip("transformers")
    pytest.skip("model download required; run scripts/run_pipeline.py for an end-to-end check")

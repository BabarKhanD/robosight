from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path  # noqa: E402
from src.pipeline import RoboSightPipeline  # noqa: E402
from src.utils.logger import get_logger, setup_logging  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
log = get_logger("eval")


def load_gt_depth(path: Path, scale: float) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read GT depth: {path}")
    return img.astype(np.float32) / scale


def find_gt(gt_dir: Path, stem: str) -> Optional[Path]:
    for name in (f"{stem}.npy", f"{stem}.png", f"{stem}.tiff", f"{stem}_depth.npy", f"{stem}_depth.png"):
        if (gt_dir / name).exists():
            return gt_dir / name
    return None


def pairwise_order_accuracy(pred_nearness: List[float], gt_depth: List[float], min_gap_ratio: float = 0.1) -> Tuple[int, int]:
    correct = total = 0
    n = len(pred_nearness)
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = gt_depth[i], gt_depth[j]
            if abs(gi - gj) / max(gi, gj) <= min_gap_ratio:
                continue
            total += 1
            if (pred_nearness[i] > pred_nearness[j]) == (gi < gj):
                correct += 1
    return correct, total


def affine_invariant_absrel(raw_pred: np.ndarray, gt_depth: np.ndarray, valid: np.ndarray) -> float:
    gt_disp = 1.0 / gt_depth[valid]
    p = raw_pred[valid].astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, gt_disp, rcond=None)
    aligned = np.clip(s * p + t, 1e-3, None)
    pred_depth = 1.0 / aligned
    return float(np.mean(np.abs(pred_depth - gt_depth[valid]) / gt_depth[valid]))


def evaluate_image(pipe: RoboSightPipeline, img_path: Path, gt_path: Optional[Path], gt_scale: float, rng: np.random.Generator, max_pixels: int) -> Dict:
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    result = pipe.process_frame(frame, 0, 0.0)
    rec: Dict = {
        "image": img_path.name,
        "objects": len(result.graph.objects),
        "relations": len(result.graph.relations),
        "timings_ms": {k: round(v, 2) for k, v in result.timings_ms.items()},
    }
    if gt_path is None:
        return rec
    gt = load_gt_depth(gt_path, gt_scale)
    h, w = frame.shape[:2]
    if gt.shape != (h, w):
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
    valid = np.isfinite(gt) & (gt > 0)
    if valid.sum() < 100:
        rec["gt_note"] = "insufficient valid GT pixels"
        return rec
    ys, xs = np.nonzero(valid)
    pick = rng.choice(len(ys), size=min(max_pixels, len(ys)), replace=False)
    rho = spearmanr(result.depth.raw[ys[pick], xs[pick]], -gt[ys[pick], xs[pick]]).statistic
    rec["pixel_spearman"] = None if np.isnan(rho) else round(float(rho), 4)
    rec["affine_invariant_absrel"] = round(affine_invariant_absrel(result.depth.raw, gt, valid), 4)
    near, gdep = [], []
    for det, obj in zip(result.detections, result.graph.objects):
        m = det.mask & valid if det.mask is not None else None
        if m is not None and m.sum() >= 20:
            near.append(obj.relative_nearness)
            gdep.append(float(np.median(gt[m])))
    rec["objects_with_gt"] = len(near)
    c, t = pairwise_order_accuracy(near, gdep)
    rec["pairs_correct"], rec["pairs_total"] = c, t
    if len(near) >= 3:
        r = spearmanr(near, [-g for g in gdep]).statistic
        rec["object_spearman"] = None if np.isnan(r) else round(float(r), 4)
    return rec


def aggregate(records: List[Dict]) -> Dict:
    agg: Dict = {"images": len(records)}
    agg["mean_objects"] = round(float(np.mean([r["objects"] for r in records])), 2) if records else 0
    agg["mean_total_ms"] = round(float(np.mean([r["timings_ms"]["total"] for r in records])), 2) if records else 0
    for key in ("pixel_spearman", "affine_invariant_absrel", "object_spearman"):
        vals = [r[key] for r in records if r.get(key) is not None]
        if vals:
            agg[f"mean_{key}"] = round(float(np.mean(vals)), 4)
            agg[f"n_{key}"] = len(vals)
    c = sum(r.get("pairs_correct", 0) for r in records)
    t = sum(r.get("pairs_total", 0) for r in records)
    if t:
        agg["object_pair_order_accuracy"] = round(c / t, 4)
        agg["object_pairs_total"] = t
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline evaluation of relative depth ordering against ground-truth depth")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--gt-dir", default=None, help="GT depth files named <stem>.npy/.png (metric depth)")
    p.add_argument("--gt-scale", type=float, default=1.0, help="divide PNG GT values by this to get meters (e.g. 1000 for mm)")
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--max-pixels", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default="results/eval_report.json")
    args = p.parse_args()

    cfg = load_config(args.config, {"runtime": {"device": args.device}})
    setup_logging(cfg["runtime"]["log_level"])
    images = sorted(f for f in Path(args.images_dir).iterdir() if f.suffix.lower() in IMAGE_EXTS)[: args.max_images]
    if not images:
        sys.exit(f"No images found in {args.images_dir}")
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    if gt_dir is None:
        log.warning("No --gt-dir given: accuracy metrics skipped; reporting counts and latency only.")

    pipe = RoboSightPipeline(cfg, tracking_enabled=False)
    rng = np.random.default_rng(args.seed)
    records = []
    for img in images:
        gt = find_gt(gt_dir, img.stem) if gt_dir else None
        if gt_dir and gt is None:
            log.warning("No GT for %s", img.name)
        records.append(evaluate_image(pipe, img, gt, args.gt_scale, rng, args.max_pixels))
    report = {"aggregate": aggregate(records), "per_image": records,
              "notes": "Metrics are computed on the supplied data only. Depth is relative; AbsRel uses per-image affine alignment in inverse-depth space."}
    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()

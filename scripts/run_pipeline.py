from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path  # noqa: E402
from src.pipeline import RoboSightPipeline  # noqa: E402
from src.reasoning.query_engine import QueryEngine  # noqa: E402
from src.utils.logger import get_logger, setup_logging  # noqa: E402
from src.utils.visualizer import draw_hud  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
log = get_logger("run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RoboSight: monocular RGB -> 2.5D spatial scene graph")
    p.add_argument("--input", required=True, help="image path, video path, or 'webcam' / camera index")
    p.add_argument("--config", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None, help="auto | cpu | cuda")
    p.add_argument("--weights", default=None)
    p.add_argument("--depth-model", default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--query", action="append", default=[], help="question for the final frame; repeatable")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--show", action="store_true", help="display a live window (press q to quit)")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Dict:
    overrides: Dict = {
        "runtime": {"device": args.device},
        "detector": {"weights": args.weights, "conf": args.conf},
        "depth": {"model_id": args.depth_model},
        "output": {"dir": args.output_dir},
    }
    cfg = load_config(args.config, overrides)
    if args.no_video:
        cfg["output"]["save_video"] = False
    return cfg


def summarize(timings: List[Dict[str, float]]) -> None:
    if not timings:
        return
    keys = ["detect", "depth", "track", "spatial", "total"]
    means = {k: sum(t[k] for t in timings) / len(timings) for k in keys}
    log.info("Measured mean latency over %d frame(s) on this machine (ms): %s",
             len(timings), ", ".join(f"{k}={v:.1f}" for k, v in means.items()))
    log.info("Effective throughput: %.2f FPS", 1000.0 / means["total"] if means["total"] else 0.0)


def answer_queries(queries: List[str], graph) -> None:
    engine = QueryEngine(graph)
    for q in queries:
        res = engine.ask(q)
        print(f"Q: {q}\nA: {res.answer}")


def run_image(pipe: RoboSightPipeline, path: Path, out_dir: Path, queries: List[str]) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    result = pipe.process_frame(frame, 0, 0.0)
    vis = pipe.render(frame, result)
    stem = path.stem
    cv2.imwrite(str(out_dir / f"{stem}_annotated.png"), vis)
    (out_dir / f"{stem}_scene_graph.json").write_text(result.graph.to_json(), encoding="utf-8")
    log.info("Wrote %s and %s", out_dir / f"{stem}_annotated.png", out_dir / f"{stem}_scene_graph.json")
    log.info("%d object(s), %d relation(s)", len(result.graph.objects), len(result.graph.relations))
    summarize([result.timings_ms])
    answer_queries(queries, result.graph)


def run_video(pipe: RoboSightPipeline, source, name: str, out_dir: Path, args: argparse.Namespace, live: bool) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cfg_out = pipe.config["output"]
    writer = None
    jsonl = open(out_dir / f"{name}_scene_graphs.jsonl", "w", encoding="utf-8") if cfg_out["save_json"] else None
    timings: List[Dict[str, float]] = []
    last_graph = None
    idx, processed, t_start = 0, 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.stride != 0:
                idx += 1
                continue
            ts = (time.time() - t_start) if live else idx / fps
            result = pipe.process_frame(frame, idx, ts)
            timings.append(result.timings_ms)
            last_graph = result.graph
            vis = pipe.render(frame, result)
            draw_hud(vis, [f"frame {idx}  objects {len(result.graph.objects)}  {result.timings_ms['total']:.0f} ms"])
            if jsonl:
                jsonl.write(result.graph.to_json(indent=None) + "\n")
            if cfg_out["save_video"]:
                if writer is None:
                    h, w = vis.shape[:2]
                    path = str(out_dir / f"{name}_annotated.mp4")
                    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), max(fps / args.stride, 1.0), (w, h))
                writer.write(vis)
            if args.show:
                cv2.imshow("RoboSight", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            processed += 1
            idx += 1
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if jsonl:
            jsonl.close()
        if args.show:
            cv2.destroyAllWindows()
    log.info("Processed %d frame(s). Outputs in %s", processed, out_dir)
    summarize(timings)
    if last_graph is not None:
        answer_queries(args.query, last_graph)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    setup_logging(cfg["runtime"]["log_level"])
    out_dir = resolve_path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    src = args.input
    is_cam = src.lower() == "webcam" or src.isdigit()
    is_image = not is_cam and Path(src).suffix.lower() in IMAGE_EXTS
    if not is_cam and not Path(src).exists():
        sys.exit(f"Input not found: {src}")
    pipe = RoboSightPipeline(cfg, tracking_enabled=False if is_image else None)
    if is_image:
        run_image(pipe, Path(src), out_dir, args.query)
    elif is_cam:
        run_video(pipe, int(src) if src.isdigit() else 0, "webcam", out_dir, args, live=True)
    else:
        run_video(pipe, src, Path(src).stem, out_dir, args, live=False)


if __name__ == "__main__":
    main()

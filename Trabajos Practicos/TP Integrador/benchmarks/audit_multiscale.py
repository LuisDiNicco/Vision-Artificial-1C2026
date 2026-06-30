"""Audita cuanto aporta la segunda escala sin ejecutar ArcFace."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tp_integrador.backend.deteccion import MediaPipeFaceDetector  # noqa: E402
from tp_integrador.backend.offline_video_analysis import (  # noqa: E402
    OfflineVideoConfig,
    _measure_quality,
    _needs_secondary_scale,
    _nms_detections,
)
from tp_integrador.backend.deteccion import FaceDetection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int)
    args = parser.parse_args()
    config = OfflineVideoConfig(secondary_strategy="adaptive")
    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frames:
        total = min(total, args.frames)
    primary = MediaPipeFaceDetector(
        max_faces=config.max_faces,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=0.60,
    )
    secondary = MediaPipeFaceDetector(
        max_faces=config.max_faces,
        min_detection_confidence=max(0.50, config.min_detection_confidence - 0.08),
        min_tracking_confidence=0.55,
    )
    stats = Counter()
    lost_examples = []
    started = time.perf_counter()
    try:
        for frame_index in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            primary_detections = list(primary.detect(frame))
            primary_kept = _nms_detections(
                primary_detections,
                config.nms_iou,
                config.max_faces,
            )
            scale = config.secondary_scale
            scaled = cv2.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            all_detections = list(primary_detections)
            for detection in secondary.detect(scaled):
                landmarks = detection.landmarks.copy()
                landmarks[:, :2] /= scale
                x1, y1, x2, y2 = detection.bbox
                all_detections.append(
                    FaceDetection(
                        bbox=(
                            int(x1 / scale),
                            int(y1 / scale),
                            int(x2 / scale),
                            int(y2 / scale),
                        ),
                        landmarks=landmarks,
                        confidence=detection.confidence,
                    )
                )
            combined = _nms_detections(all_detections, config.nms_iou, config.max_faces)
            run_adaptive = _needs_secondary_scale(
                primary_detections,
                config,
                frame_index,
                scene_cut=False,
            )
            adaptive = combined if run_adaptive else primary_kept
            stats["frames"] += 1
            stats["primary_faces"] += len(primary_kept)
            stats["always_faces"] += len(combined)
            stats["adaptive_faces"] += len(adaptive)
            stats["adaptive_secondary_frames"] += int(run_adaptive)
            lost = max(0, len(combined) - len(adaptive))
            stats["lost_faces"] += lost
            if lost:
                stats["frames_with_loss"] += 1
                extras = combined[len(primary_kept) :]
                usable = sum(_measure_quality(frame, face, config).usable for face in extras)
                stats["lost_usable_faces_upper_bound"] += usable
                if len(lost_examples) < 20:
                    lost_examples.append(
                        {
                            "frame": frame_index,
                            "primary": len(primary_kept),
                            "always": len(combined),
                            "lost": lost,
                            "usable_upper_bound": usable,
                        }
                    )
    finally:
        cap.release()
        primary.close()
        secondary.close()
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                **stats,
                "wall_seconds": elapsed,
                "adaptive_secondary_percent": 100.0
                * stats["adaptive_secondary_frames"]
                / max(stats["frames"], 1),
                "face_recall_vs_always_percent": 100.0
                * stats["adaptive_faces"]
                / max(stats["always_faces"], 1),
                "lost_examples": lost_examples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

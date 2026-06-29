from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .clasificador import Prediction
from .celebrity import CELEBRITY_INDEX_PATH, CELEBRITY_MIN_MARGIN
from .deteccion import FaceDetection
from .video_inputs import VIDEO_DOWNLOAD_DIR


ANALYSIS_CACHE_DIR = VIDEO_DOWNLOAD_DIR / "analysis"
ANALYSIS_CACHE_VERSION = 6
LANDMARK_CACHE_STEP = 1


def analysis_cache_path(video_path: Path, sample_seconds: float, min_similarity: float) -> Path:
    video_path = video_path.resolve()
    stat = video_path.stat()
    identity = {
        "version": ANALYSIS_CACHE_VERSION,
        "path": str(video_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sample_seconds": round(float(sample_seconds), 4),
        "min_similarity": round(float(min_similarity), 4),
        "min_similarity_margin": CELEBRITY_MIN_MARGIN,
    }
    if CELEBRITY_INDEX_PATH.exists():
        celebrity_stat = CELEBRITY_INDEX_PATH.stat()
        identity["celebrity_index_size"] = celebrity_stat.st_size
        identity["celebrity_index_mtime_ns"] = celebrity_stat.st_mtime_ns
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return ANALYSIS_CACHE_DIR / f"{digest}.json"


def save_video_analysis(
    path: Path,
    video_path: Path,
    duration: float,
    fps: float,
    width: int,
    height: int,
    sample_seconds: float,
    min_similarity: float,
    records: Iterable[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": ANALYSIS_CACHE_VERSION,
        "video_path": str(video_path.resolve()),
        "duration": float(duration),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "sample_seconds": float(sample_seconds),
        "min_similarity": float(min_similarity),
        "records": list(records),
    }
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temp_path.replace(path)


def load_video_analysis(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != ANALYSIS_CACHE_VERSION:
        raise ValueError("Version de cache de video incompatible.")
    return payload


def make_analysis_record(
    seconds: float,
    detections: Iterable[FaceDetection],
    predictions: Iterable[Prediction],
) -> dict:
    detections = list(detections)
    predictions = list(predictions)
    faces = []
    for index, detection in enumerate(detections):
        prediction = predictions[index] if index < len(predictions) else None
        faces.append(
            {
                "bbox": [int(value) for value in detection.bbox],
                "landmarks": [
                    [int(round(point[0])), int(round(point[1])), round(float(point[2]), 6)]
                    for point in detection.landmarks[::LANDMARK_CACHE_STEP]
                ],
                "detection_confidence": float(detection.confidence),
                "label": prediction.label if prediction else "desconocido",
                "confidence": float(prediction.confidence) if prediction else 0.0,
                "distance": float(prediction.distance)
                if prediction and math.isfinite(float(prediction.distance))
                else None,
                "method": prediction.method if prediction else "sin_prediccion",
            }
        )
    return {"seconds": float(seconds), "faces": faces}


def prepare_analysis_timeline(payload: dict) -> tuple[list[float], list[dict]]:
    records = payload.get("records", [])
    return [float(record["seconds"]) for record in records], records


def analysis_at_time(times: list[float], records: list[dict], seconds: float):
    if not records:
        return [], []
    index = max(0, bisect.bisect_right(times, seconds) - 1)
    faces = records[index].get("faces", [])
    detections = []
    predictions = []
    for face in faces:
        landmark_points = np.array(
            [
                [point[0], point[1], point[2] if len(point) > 2 else 0.0]
                for point in face.get("landmarks", [])
            ],
            dtype=np.float32,
        )
        if landmark_points.size == 0:
            landmark_points = np.empty((0, 3), dtype=np.float32)
        detections.append(
            FaceDetection(
                bbox=tuple(int(value) for value in face["bbox"]),
                landmarks=landmark_points,
                confidence=float(face.get("detection_confidence", 1.0)),
                landmarks_are_sampled=True,
            )
        )
        predictions.append(
            Prediction(
                label=str(face.get("label", "desconocido")),
                confidence=float(face.get("confidence", 0.0)),
                distance=float(face["distance"]) if face.get("distance") is not None else float("inf"),
                method=str(face.get("method", "cache")),
            )
        )
    return detections, predictions

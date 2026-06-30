from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .clasificador import Prediction
from .celebrity import CELEBRITY_INDEX_PATH, CELEBRITY_MIN_MARGIN
from .deteccion import FaceDetection
from .video_inputs import VIDEO_DOWNLOAD_DIR


ANALYSIS_CACHE_DIR = VIDEO_DOWNLOAD_DIR / "analysis"
FEATURE_CACHE_DIR = ANALYSIS_CACHE_DIR / "features"
ANALYSIS_CACHE_VERSION = 12
VIDEO_FEATURE_CACHE_VERSION = 1


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


def video_feature_cache_path(video_path: Path) -> Path:
    """Cachea detecciones y embeddings, independientes de umbrales de identidad."""
    video_path = video_path.resolve()
    stat = video_path.stat()
    identity = {
        "version": VIDEO_FEATURE_CACHE_VERSION,
        "path": str(video_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return FEATURE_CACHE_DIR / f"{digest}.npz"


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
    analysis_metadata: dict | None = None,
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
        "analysis_metadata": analysis_metadata or {},
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


def prepare_analysis_timeline(payload: dict) -> tuple[list[float], list[dict]]:
    records = payload.get("records", [])
    return [float(record["seconds"]) for record in records], records


def analysis_record_at_time(times: list[float], records: list[dict], seconds: float) -> dict | None:
    """Devuelve el registro offline mas cercano para mostrar su evidencia sin interpolarla."""
    if not records:
        return None
    next_index = bisect.bisect_left(times, seconds)
    if next_index <= 0:
        return records[0]
    if next_index >= len(records):
        return records[-1]
    previous_index = next_index - 1
    if seconds - times[previous_index] <= times[next_index] - seconds:
        return records[previous_index]
    return records[next_index]


def analysis_at_time(times: list[float], records: list[dict], seconds: float):
    if not records:
        return [], []
    if seconds <= times[0]:
        return _faces_to_outputs(records[0].get("faces", []))
    if seconds >= times[-1]:
        return _faces_to_outputs(records[-1].get("faces", []))

    next_index = max(1, bisect.bisect_right(times, seconds))
    prev_index = next_index - 1
    prev_time = times[prev_index]
    next_time = times[next_index]
    if next_time <= prev_time + 1e-6:
        return _faces_to_outputs(records[prev_index].get("faces", []))

    alpha = float(np.clip((seconds - prev_time) / (next_time - prev_time), 0.0, 1.0))
    if alpha <= 0.0:
        return _faces_to_outputs(records[prev_index].get("faces", []))
    if alpha >= 1.0:
        return _faces_to_outputs(records[next_index].get("faces", []))

    prev_faces = list(records[prev_index].get("faces", []))
    next_faces = list(records[next_index].get("faces", []))
    detections = []
    predictions = []
    matched_next = set()

    for prev_face in prev_faces:
        match_index, match_score = _best_face_match(prev_face, next_faces, matched_next)
        if match_index is None or match_score < 0.18:
            if alpha < 0.5:
                detection, prediction = _face_to_outputs(prev_face)
                detections.append(detection)
                predictions.append(prediction)
            continue

        matched_next.add(match_index)
        next_face = next_faces[match_index]
        detection, prediction = _blend_face_outputs(prev_face, next_face, alpha)
        detections.append(detection)
        predictions.append(prediction)

    for next_index_face, next_face in enumerate(next_faces):
        if next_index_face in matched_next:
            continue
        if alpha >= 0.5:
            detection, prediction = _face_to_outputs(next_face)
            detections.append(detection)
            predictions.append(prediction)

    return detections, predictions


def _faces_to_outputs(faces: list[dict]):
    detections = []
    predictions = []
    for face in faces:
        detection, prediction = _face_to_outputs(face)
        detections.append(detection)
        predictions.append(prediction)
    return detections, predictions


def _face_to_outputs(face: dict) -> tuple[FaceDetection, Prediction]:
    landmark_points = _cached_landmarks_to_array(face.get("landmarks", []))
    detection = FaceDetection(
        bbox=tuple(int(value) for value in face["bbox"]),
        landmarks=landmark_points,
        confidence=float(face.get("detection_confidence", 1.0)),
        landmarks_are_sampled=True,
    )
    prediction = Prediction(
        label=str(face.get("label", "desconocido")),
        confidence=float(face.get("confidence", 0.0)),
        distance=float(face["distance"]) if face.get("distance") is not None else float("inf"),
        method=str(face.get("method", "cache")),
    )
    return detection, prediction


def _blend_face_outputs(prev_face: dict, next_face: dict, alpha: float) -> tuple[FaceDetection, Prediction]:
    prev_detection, prev_prediction = _face_to_outputs(prev_face)
    next_detection, next_prediction = _face_to_outputs(next_face)

    bbox = tuple(
        int(round((1.0 - alpha) * float(prev_value) + alpha * float(next_value)))
        for prev_value, next_value in zip(prev_detection.bbox, next_detection.bbox)
    )

    if (
        prev_detection.landmarks.size
        and next_detection.landmarks.size
        and prev_detection.landmarks.shape == next_detection.landmarks.shape
    ):
        landmarks = (1.0 - alpha) * prev_detection.landmarks + alpha * next_detection.landmarks
    else:
        landmarks = prev_detection.landmarks if alpha < 0.5 else next_detection.landmarks

    if prev_prediction.label == next_prediction.label:
        label = prev_prediction.label
        confidence = float((1.0 - alpha) * prev_prediction.confidence + alpha * next_prediction.confidence)
        if np.isfinite(prev_prediction.distance) and np.isfinite(next_prediction.distance):
            distance = float((1.0 - alpha) * prev_prediction.distance + alpha * next_prediction.distance)
        else:
            distance = prev_prediction.distance if alpha < 0.5 else next_prediction.distance
        method = prev_prediction.method if alpha < 0.5 else next_prediction.method
    elif alpha < 0.5:
        label = prev_prediction.label
        confidence = prev_prediction.confidence
        distance = prev_prediction.distance
        method = prev_prediction.method
    else:
        label = next_prediction.label
        confidence = next_prediction.confidence
        distance = next_prediction.distance
        method = next_prediction.method

    detection = FaceDetection(
        bbox=bbox,
        landmarks=landmarks.astype(np.float32, copy=False),
        confidence=float((1.0 - alpha) * prev_detection.confidence + alpha * next_detection.confidence),
        landmarks_are_sampled=True,
    )
    prediction = Prediction(
        label=label,
        confidence=float(confidence),
        distance=float(distance),
        method=method,
    )
    return detection, prediction


def _cached_landmarks_to_array(landmarks: list[list[float]]) -> np.ndarray:
    landmark_points = np.array(
        [
            [point[0], point[1], point[2] if len(point) > 2 else 0.0]
            for point in landmarks
        ],
        dtype=np.float32,
    )
    if landmark_points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return landmark_points


def _best_face_match(face: dict, candidates: list[dict], used: set[int]) -> tuple[int | None, float]:
    best_index = None
    best_score = 0.0
    for index, candidate in enumerate(candidates):
        if index in used:
            continue
        score = _cached_face_match_score(face.get("bbox", (0, 0, 0, 0)), candidate.get("bbox", (0, 0, 0, 0)))
        if score > best_score:
            best_score = score
            best_index = index
    return best_index, best_score


def _cached_face_match_score(a_bbox, b_bbox) -> float:
    iou = _bbox_iou(a_bbox, b_bbox)
    ax, ay = _bbox_center(a_bbox)
    bx, by = _bbox_center(b_bbox)
    diagonal = np.sqrt(max(_bbox_area(a_bbox), _bbox_area(b_bbox))) * 1.8
    center_score = max(0.0, 1.0 - float(np.hypot(ax - bx, ay - by)) / max(diagonal, 1.0))
    return max(iou, center_score * 0.72)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return float(intersection / union) if union else 0.0

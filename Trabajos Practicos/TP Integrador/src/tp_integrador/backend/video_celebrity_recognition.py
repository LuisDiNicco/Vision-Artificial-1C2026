from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import cv2
import numpy as np

from .celebrity import (
    CELEBRITY_MIN_MARGIN,
    CelebrityIndex,
    CelebrityMatch,
    celebrity_match_rejection_reason,
)
from .deteccion import FaceDetection, MediaPipeFaceDetector
from .face_quality import VIDEO_FACE_QUALITY, assess_face_quality


ProgressCallback = Callable[[str], None]


@dataclass
class VideoCelebrityConfig:
    sample_every_seconds: float = 0.35
    max_frames: int = 220
    min_quality_score: float = 0.42
    min_face_side: int = 112
    min_track_samples: int = 3
    min_similarity: float = 0.34
    min_similarity_margin: float = CELEBRITY_MIN_MARGIN
    max_faces_per_frame: int = 6


@dataclass
class VideoCelebrityResult:
    name: str
    confidence: float
    similarity: float
    samples: int
    first_second: float
    last_second: float
    bbox: tuple[int, int, int, int]
    thumbnail: np.ndarray
    matches: List[CelebrityMatch] = field(default_factory=list)
    quality_score: float = 0.0


@dataclass
class _FaceObservation:
    frame_index: int
    second: float
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    quality_score: float
    face_side: int
    thumbnail: np.ndarray


@dataclass
class _FaceTrack:
    observations: List[_FaceObservation] = field(default_factory=list)

    @property
    def last_bbox(self) -> tuple[int, int, int, int]:
        return self.observations[-1].bbox

    def append(self, observation: _FaceObservation) -> None:
        self.observations.append(observation)


def recognize_celebrities_in_video(
    video_path: Path | str,
    detector: MediaPipeFaceDetector,
    embedder,
    celebrity_index: CelebrityIndex,
    config: Optional[VideoCelebrityConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[VideoCelebrityResult]:
    config = config or VideoCelebrityConfig()
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * config.sample_every_seconds)))

    tracks: List[_FaceTrack] = []
    processed = 0
    frame_index = 0

    try:
        while processed < config.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue

            detections = _largest_faces(detector.detect(frame), config.max_faces_per_frame)
            observations = [
                observation
                for detection in detections
                for observation in [_observation_from_detection(frame, detection, embedder, frame_index, fps, config)]
                if observation is not None
            ]
            _assign_observations_to_tracks(tracks, observations)

            processed += 1
            if progress_callback and (processed == 1 or processed % 12 == 0):
                percent = (frame_index / total_frames * 100.0) if total_frames else 0.0
                progress_callback(f"Analizando video... {processed} muestras, {percent:.0f}%")

            frame_index += 1
    finally:
        cap.release()

    if progress_callback:
        progress_callback("Comparando rostros contra famosos...")
    return _results_from_tracks(tracks, celebrity_index, config)


def compose_video_results_view(
    results: Iterable[VideoCelebrityResult],
    width: int,
    height: int,
    title: str = "Actores reconocidos en el video",
) -> np.ndarray:
    results = list(results)
    canvas = np.full((height, width, 3), (24, 27, 32), dtype=np.uint8)
    cv2.putText(canvas, title, (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (240, 244, 248), 2, cv2.LINE_AA)
    cv2.line(canvas, (28, 62), (width - 28, 62), (70, 180, 220), 2, cv2.LINE_AA)

    x = 28
    y = 90
    tile_w = min(380, max(290, (width - 84) // 3))
    tile_h = 166
    for rank, result in enumerate(results, start=1):
        if y + tile_h > height - 24:
            break
        cv2.rectangle(canvas, (x, y), (x + tile_w, y + tile_h), (38, 43, 51), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, y), (x + tile_w, y + tile_h), (70, 180, 220), 1, cv2.LINE_AA)

        thumb = cv2.resize(result.thumbnail, (118, 118), interpolation=cv2.INTER_AREA)
        canvas[y + 18 : y + 136, x + 16 : x + 134] = thumb
        text_x = x + 150
        cv2.putText(canvas, f"{rank}. {result.name}"[:30], (text_x, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 247, 250), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"conf {result.confidence * 100:.0f}%  sim {result.similarity * 100:.1f}%", (text_x, y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (145, 210, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"muestras {result.samples}  {result.first_second:.1f}s-{result.last_second:.1f}s", (text_x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 200, 210), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"calidad {result.quality_score:.2f}", (text_x, y + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 200, 210), 1, cv2.LINE_AA)

        x += tile_w + 14
        if x + tile_w > width - 28:
            x = 28
            y += tile_h + 16

    if not results:
        cv2.putText(canvas, "No hubo coincidencias confiables.", (28, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (190, 200, 210), 1, cv2.LINE_AA)
    return canvas


def _observation_from_detection(
    frame: np.ndarray,
    detection: FaceDetection,
    embedder,
    frame_index: int,
    fps: float,
    config: VideoCelebrityConfig,
) -> Optional[_FaceObservation]:
    x1, y1, x2, y2 = detection.bbox
    face_side = min(x2 - x1, y2 - y1)
    if face_side < config.min_face_side:
        return None
    quality = assess_face_quality(frame, detection, VIDEO_FACE_QUALITY)
    if not quality.ok or quality.score < config.min_quality_score:
        return None
    embedding, aligned = embedder.embed_face(frame, detection)
    return _FaceObservation(
        frame_index=frame_index,
        second=frame_index / max(fps, 1e-6),
        bbox=detection.bbox,
        embedding=embedding,
        quality_score=quality.score,
        face_side=face_side,
        thumbnail=aligned,
    )


def _assign_observations_to_tracks(tracks: List[_FaceTrack], observations: List[_FaceObservation]) -> None:
    for observation in observations:
        best_track = None
        best_score = 0.0
        for track in tracks:
            score = _track_match_score(track.last_bbox, observation.bbox)
            if score > best_score:
                best_track = track
                best_score = score
        if best_track is not None and best_score >= 0.28:
            best_track.append(observation)
        else:
            tracks.append(_FaceTrack([observation]))


def _results_from_tracks(
    tracks: List[_FaceTrack],
    celebrity_index: CelebrityIndex,
    config: VideoCelebrityConfig,
) -> List[VideoCelebrityResult]:
    results: List[VideoCelebrityResult] = []
    for track in tracks:
        if len(track.observations) < config.min_track_samples:
            continue
        embeddings = np.vstack([item.embedding for item in track.observations]).astype(np.float32)
        weights = np.array([item.quality_score * np.sqrt(item.face_side) for item in track.observations], dtype=np.float32)
        embedding = np.average(embeddings, axis=0, weights=weights)
        embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-6)
        matches = celebrity_index.top_unique(embedding.astype(np.float32), limit=5)
        if not matches:
            continue
        best = matches[0]
        if celebrity_match_rejection_reason(
            matches,
            config.min_similarity,
            config.min_similarity_margin,
        ) is not None:
            continue
        best_observation = max(track.observations, key=lambda item: (item.quality_score, item.face_side))
        confidence = _confidence(best.similarity, len(track.observations), config)
        results.append(
            VideoCelebrityResult(
                name=best.name,
                confidence=confidence,
                similarity=best.similarity,
                samples=len(track.observations),
                first_second=track.observations[0].second,
                last_second=track.observations[-1].second,
                bbox=best_observation.bbox,
                thumbnail=best_observation.thumbnail,
                matches=matches,
                quality_score=float(np.mean([item.quality_score for item in track.observations])),
            )
        )

    results.sort(key=lambda item: (item.confidence, item.similarity, item.samples), reverse=True)
    return _deduplicate_names(results)


def _deduplicate_names(results: List[VideoCelebrityResult]) -> List[VideoCelebrityResult]:
    unique = {}
    for result in results:
        current = unique.get(result.name)
        if current is None or (result.confidence, result.samples) > (current.confidence, current.samples):
            unique[result.name] = result
    return sorted(unique.values(), key=lambda item: (item.confidence, item.similarity), reverse=True)


def _largest_faces(detections: Iterable[FaceDetection], limit: int) -> List[FaceDetection]:
    return sorted(detections, key=lambda item: _bbox_area(item.bbox), reverse=True)[:limit]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _track_match_score(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    iou = _bbox_iou(a, b)
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    diagonal = np.sqrt(max(_bbox_area(a), _bbox_area(b))) * 1.8
    center_score = max(0.0, 1.0 - float(np.hypot(ax - bx, ay - by)) / max(diagonal, 1.0))
    return max(iou, center_score * 0.72)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


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


def _confidence(similarity: float, samples: int, config: VideoCelebrityConfig) -> float:
    similarity_score = np.clip((similarity - config.min_similarity) / 0.28, 0.0, 1.0)
    evidence_score = np.clip(samples / max(config.min_track_samples * 2.5, 1.0), 0.0, 1.0)
    return float(0.72 * similarity_score + 0.28 * evidence_score)

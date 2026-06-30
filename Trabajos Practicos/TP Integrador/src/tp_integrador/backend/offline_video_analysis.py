from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .alineamiento import eye_centers
from .celebrity import (
    CELEBRITY_MIN_MARGIN,
    CelebrityIndex,
    CelebrityMatch,
    celebrity_match_rejection_reason,
)
from .deteccion import FaceDetection, MediaPipeFaceDetector


ProgressCallback = Callable[[float, str], None]
NOSE_TIP = 1
CHIN = 152
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
FEATURE_CACHE_VERSION = 1


@dataclass
class OfflineVideoConfig:
    max_frames: Optional[int] = None
    secondary_scale: float = 1.5
    max_faces: int = 10
    min_detection_confidence: float = 0.65
    nms_iou: float = 0.68
    min_face_side: int = 64
    min_eye_distance: float = 20.0
    max_track_gap: int = 12
    max_fill_gap: int = 15
    min_track_embeddings: int = 3
    min_similarity: float = 0.34
    min_similarity_margin: float = CELEBRITY_MIN_MARGIN
    min_temporal_votes: int = 3
    temporal_vote_ratio: float = 0.65
    strong_track_min_vote_ratio: float = 0.45
    max_competing_vote_ratio: float = 0.15
    min_embedding_inlier_ratio: float = 0.68
    sample_support_count: int = 2
    sample_support_slack: float = 0.05
    max_refinement_frames: int = 5
    embedding_batch_size: int = 12
    max_batch_frames: int = 24
    secondary_strategy: str = "always"
    secondary_scan_interval: int = 20
    secondary_trigger_side: int = 180
    parallel_pipeline: bool = True
    confidence_similarity_span: float = 0.28
    confidence_margin_span: float = 0.12
    known_state_persistence: float = 0.98
    unknown_state_persistence: float = 0.93
    frame_known_probability: float = 0.35
    accepted_track_probability_floor: float = 0.45
    min_competing_segment_frames: int = 8
    competing_segment_ratio: float = 0.80
    global_confidence_weight: float = 0.70
    frame_confidence_weight: float = 0.30


@dataclass
class OfflineVideoResult:
    duration: float
    fps: float
    width: int
    height: int
    records: list[dict]
    metadata: dict


@dataclass
class _Quality:
    usable: bool
    weight: float
    reason: str
    side: int
    eye_distance: float
    brightness: float
    blur: float
    nose_shift: float

    def as_dict(self) -> dict:
        return {
            "usable": self.usable,
            "weight": round(self.weight, 5),
            "reason": self.reason,
            "side": self.side,
            "eye_distance": round(self.eye_distance, 3),
            "brightness": round(self.brightness, 3),
            "blur": round(self.blur, 3),
            "nose_shift": round(self.nose_shift, 5),
        }


@dataclass
class _Observation:
    frame_index: int
    second: float
    detection: FaceDetection
    quality: _Quality
    embedding: Optional[np.ndarray]
    top_matches: list[CelebrityMatch] = field(default_factory=list)
    track_id: int = -1
    synthetic: bool = False
    scene_id: int = 0
    alternate_embedding: Optional[np.ndarray] = None


@dataclass
class _TrackDecision:
    label: str = "desconocido"
    confidence: float = 0.0
    similarity: float = 0.0
    distance: float = float("inf")
    top_matches: list[CelebrityMatch] = field(default_factory=list)
    support_count: int = 0
    votes: int = 0
    competing_votes: int = 0
    evidence_frames: int = 0
    vote_ratio: float = 0.0
    inlier_ratio: float = 0.0
    rejection_reason: str = ""


@dataclass
class _FrameDecision:
    label: str = "desconocido"
    confidence: float = 0.0
    distance: float = float("inf")
    similarity: float = 0.0
    margin: float = 0.0
    local_confidence: float = 0.0
    temporal_probability: float = 0.0
    reason: str = "sin_evidencia"


@dataclass
class _Track:
    track_id: int
    observations: list[_Observation] = field(default_factory=list)
    extra_embeddings: list[np.ndarray] = field(default_factory=list)
    decision: _TrackDecision = field(default_factory=_TrackDecision)

    @property
    def last(self) -> _Observation:
        return self.observations[-1]


def analyze_video_offline(
    video_path: Path | str,
    embedder,
    celebrity_index: CelebrityIndex,
    min_similarity: float = 0.34,
    cancel_event=None,
    progress_callback: Optional[ProgressCallback] = None,
    config: Optional[OfflineVideoConfig] = None,
    feature_cache_path: Optional[Path | str] = None,
) -> OfflineVideoResult:
    config = config or OfflineVideoConfig(min_similarity=min_similarity)
    config.min_similarity = min_similarity
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video para preprocesarlo.")

    fps = max(float(cap.get(cv2.CAP_PROP_FPS) or 25.0), 1.0)
    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("El video no informa una cantidad valida de frames.")
    if config.max_frames is not None:
        total_frames = min(total_frames, max(1, config.max_frames))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    feature_cache_path = Path(feature_cache_path) if feature_cache_path else None
    if feature_cache_path is not None and feature_cache_path.exists() and config.max_frames is None:
        try:
            feature_metadata, feature_arrays = _load_feature_cache(feature_cache_path)
            frame_observations = _observations_from_feature_arrays(feature_arrays, total_frames, fps)
        except Exception:
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError("No se pudo reabrir el video tras descartar un cache invalido.")
        else:
            cap.release()
            tracks: list[_Track] = []
            for observations in frame_observations:
                for observation in observations:
                    if observation.embedding is not None:
                        observation.top_matches = celebrity_index.top_unique(
                            observation.embedding,
                            limit=3,
                        )
                _assign_tracks(tracks, observations, config)
            stats = Counter(feature_metadata.get("stats", {}))
            return _finalize_analysis(
                video_path,
                embedder,
                celebrity_index,
                config,
                cancel_event,
                progress_callback,
                duration,
                fps,
                width,
                height,
                frame_observations,
                tracks,
                stats,
                feature_cache_path=None,
                reused_features=True,
            )

    primary_detector = MediaPipeFaceDetector(
        max_faces=config.max_faces,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=0.60,
    )
    secondary_detector = MediaPipeFaceDetector(
        max_faces=config.max_faces,
        min_detection_confidence=max(0.50, config.min_detection_confidence - 0.08),
        min_tracking_confidence=0.55,
    )
    frame_observations: list[list[_Observation]] = [[] for _ in range(total_frames)]
    tracks: list[_Track] = []
    stats = Counter()
    scene_id = 0
    previous_scene_signature = None
    pending_frames = []
    pending_embedding_count = 0
    inference_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="arcface-batch")
        if config.parallel_pipeline and config.embedding_batch_size > 1
        else None
    )
    inflight_batch = None

    try:
        for frame_index in range(total_frames):
            _check_cancel(cancel_event)
            ok, frame = cap.read()
            if not ok:
                break
            scene_signature = _scene_signature(frame)
            scene_cut = False
            if previous_scene_signature is not None and _is_scene_cut(
                previous_scene_signature,
                scene_signature,
            ):
                scene_id += 1
                scene_cut = True
                stats["scene_cuts"] += 1
            previous_scene_signature = scene_signature
            detections = _multiscale_detections(
                frame,
                primary_detector,
                secondary_detector,
                config,
                frame_index=frame_index,
                scene_cut=scene_cut or frame_index == 0,
            )
            stats["detections"] += len(detections)
            observations = []
            aligned_faces = []
            for detection in detections:
                quality = _measure_quality(frame, detection, config)
                observation = _Observation(
                    frame_index=frame_index,
                    second=frame_index / fps,
                    detection=detection,
                    quality=quality,
                    embedding=None,
                    scene_id=scene_id,
                )
                observations.append(observation)
                if quality.usable and config.embedding_batch_size > 1:
                    try:
                        aligned_faces.append((observation, embedder.align_face_input(frame, detection)))
                    except Exception:
                        quality.reason = _append_reason(quality.reason, "embedding_error")
                        stats["embedding_errors"] += 1
                elif quality.usable:
                    try:
                        observation.embedding = _tta_embedding(frame, detection, embedder)
                    except Exception:
                        quality.reason = _append_reason(quality.reason, "embedding_error")
                        stats["embedding_errors"] += 1
                else:
                    stats[f"hard_reject:{quality.reason}"] += 1
            pending_frames.append((frame_index, observations, aligned_faces))
            pending_embedding_count += len(aligned_faces)
            should_flush = (
                config.embedding_batch_size <= 1
                or pending_embedding_count >= config.embedding_batch_size
                or len(pending_frames) >= config.max_batch_frames
                or frame_index + 1 == total_frames
            )
            if should_flush:
                current_batch = list(pending_frames)
                pending_frames.clear()
                pending_embedding_count = 0
                if inference_executor is not None:
                    if inflight_batch is not None:
                        _complete_pending_frames(
                            *inflight_batch,
                            embedder,
                            celebrity_index,
                            tracks,
                            frame_observations,
                            config,
                            stats,
                        )
                    aligned = [face for _, _, items in current_batch for _, face in items]
                    future = inference_executor.submit(embedder.embed_tta_batch, aligned)
                    inflight_batch = (current_batch, future)
                else:
                    _flush_pending_frames(
                        current_batch,
                        embedder,
                        celebrity_index,
                        tracks,
                        frame_observations,
                        config,
                        stats,
                    )

                # El vaciado ocurre por cantidad de embeddings o por bloques de
                # frames. Sus indices no necesariamente coinciden con multiplos
                # del FPS (por ejemplo, bloques de 24 en un video de 30 FPS), por
                # lo que filtrar nuevamente por FPS podia impedir que la GUI
                # recibiera progreso durante toda la primera pasada.
                if progress_callback:
                    progress_callback(
                        0.82 * (frame_index + 1) / total_frames,
                        f"Pasada 1/2: frame {frame_index + 1}/{total_frames}",
                    )
    except BaseException:
        if inference_executor is not None:
            inference_executor.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        cap.release()
        primary_detector.close()
        secondary_detector.close()

    if inflight_batch is not None:
        _complete_pending_frames(
            *inflight_batch,
            embedder,
            celebrity_index,
            tracks,
            frame_observations,
            config,
            stats,
        )
    if inference_executor is not None:
        inference_executor.shutdown(wait=True)

    if pending_frames:
        _flush_pending_frames(
            pending_frames,
            embedder,
            celebrity_index,
            tracks,
            frame_observations,
            config,
            stats,
        )

    return _finalize_analysis(
        video_path,
        embedder,
        celebrity_index,
        config,
        cancel_event,
        progress_callback,
        duration,
        fps,
        width,
        height,
        frame_observations,
        tracks,
        stats,
        feature_cache_path=feature_cache_path if config.max_frames is None else None,
        reused_features=False,
    )


def _finalize_analysis(
    video_path,
    embedder,
    celebrity_index,
    config,
    cancel_event,
    progress_callback,
    duration,
    fps,
    width,
    height,
    frame_observations,
    tracks,
    stats,
    feature_cache_path,
    reused_features,
):
    if progress_callback:
        message = (
            f"Reutilizando caracteristicas de {len(tracks)} tracks"
            if reused_features
            else f"Pasada 2/2: consolidando {len(tracks)} tracks"
        )
        progress_callback(0.83, message)

    tracks = _merge_compatible_tracks(tracks, config)
    for track in tracks:
        track.decision = _decide_track(track, celebrity_index, config)

    if not reused_features:
        unresolved = [track for track in tracks if track.decision.label == "desconocido"]
        _refine_unresolved_tracks(
            video_path,
            unresolved,
            embedder,
            celebrity_index,
            config,
            cancel_event,
            progress_callback,
        )

    for track in tracks:
        if track.extra_embeddings or any(
            item.alternate_embedding is not None for item in track.observations
        ):
            track.decision = _decide_track(track, celebrity_index, config)

    if feature_cache_path is not None:
        arrays = _observations_to_feature_arrays(frame_observations)
        _save_feature_cache(
            feature_cache_path,
            {
                "version": FEATURE_CACHE_VERSION,
                "frames": len(frame_observations),
                "fps": fps,
                "width": width,
                "height": height,
                "stats": {str(key): int(value) for key, value in stats.items()},
            },
            arrays,
        )

    synthetic_count = _fill_short_track_gaps(frame_observations, tracks, config, fps)
    records = _records_from_observations(frame_observations, tracks, fps, config)
    recognized_tracks = sum(track.decision.label != "desconocido" for track in tracks)
    rejection_reasons = Counter(
        track.decision.rejection_reason
        for track in tracks
        if track.decision.label == "desconocido" and track.decision.rejection_reason
    )
    metadata = {
        "pipeline": "offline_bidirectional_tracks_v4_cpu_batch",
        "frames_analyzed": len(frame_observations),
        "tracks": len(tracks),
        "recognized_tracks": recognized_tracks,
        "rejection_reasons": dict(rejection_reasons),
        "synthetic_gap_frames": synthetic_count,
        "detections": stats["detections"],
        "embedding_errors": stats["embedding_errors"],
        "scene_cuts": stats["scene_cuts"],
        "min_similarity": config.min_similarity,
        "min_similarity_margin": config.min_similarity_margin,
        "temporal_vote_ratio": config.temporal_vote_ratio,
        "strong_track_min_vote_ratio": config.strong_track_min_vote_ratio,
        "max_competing_vote_ratio": config.max_competing_vote_ratio,
        "secondary_scale": config.secondary_scale,
        "secondary_strategy": config.secondary_strategy,
        "embedding_batch_size": config.embedding_batch_size,
        "parallel_pipeline": config.parallel_pipeline,
        "features_reused": reused_features,
        "min_face_side": config.min_face_side,
        "nms_iou": config.nms_iou,
    }
    if progress_callback:
        progress_callback(1.0, "Analisis offline consolidado")
    return OfflineVideoResult(duration, fps, width, height, records, metadata)


def _observations_to_feature_arrays(frame_observations):
    observations = [item for frame in frame_observations for item in frame if not item.synthetic]
    count = len(observations)
    landmarks_shape = observations[0].detection.landmarks.shape if observations else (478, 3)
    embeddings = np.full((count, 512), np.nan, dtype=np.float32)
    alternates = np.full((count, 512), np.nan, dtype=np.float32)
    landmarks = np.empty((count, *landmarks_shape), dtype=np.float32)
    for index, observation in enumerate(observations):
        landmarks[index] = observation.detection.landmarks
        if observation.embedding is not None:
            embeddings[index] = observation.embedding
        if observation.alternate_embedding is not None:
            alternates[index] = observation.alternate_embedding
    return {
        "frame_index": np.asarray([item.frame_index for item in observations], dtype=np.int32),
        "scene_id": np.asarray([item.scene_id for item in observations], dtype=np.int32),
        "bbox": np.asarray([item.detection.bbox for item in observations], dtype=np.int32).reshape(count, 4),
        "landmarks": landmarks,
        "detection_confidence": np.asarray(
            [item.detection.confidence for item in observations], dtype=np.float32
        ),
        "quality_usable": np.asarray([item.quality.usable for item in observations], dtype=np.bool_),
        "quality_weight": np.asarray([item.quality.weight for item in observations], dtype=np.float32),
        "quality_reason": np.asarray([item.quality.reason for item in observations], dtype=np.str_),
        "quality_side": np.asarray([item.quality.side for item in observations], dtype=np.int32),
        "quality_metrics": np.asarray(
            [
                (
                    item.quality.eye_distance,
                    item.quality.brightness,
                    item.quality.blur,
                    item.quality.nose_shift,
                )
                for item in observations
            ],
            dtype=np.float32,
        ).reshape(count, 4),
        "embedding": embeddings,
        "alternate_embedding": alternates,
        "frame_count": np.asarray([len(frame_observations)], dtype=np.int32),
    }


def _observations_from_feature_arrays(arrays, total_frames, fps):
    cached_frames = int(arrays["frame_count"][0])
    if cached_frames != total_frames:
        raise ValueError("El cache de caracteristicas no coincide con el video.")
    frame_observations = [[] for _ in range(total_frames)]
    for index, frame_index_value in enumerate(arrays["frame_index"]):
        frame_index = int(frame_index_value)
        metrics = arrays["quality_metrics"][index]
        quality = _Quality(
            bool(arrays["quality_usable"][index]),
            float(arrays["quality_weight"][index]),
            str(arrays["quality_reason"][index]),
            int(arrays["quality_side"][index]),
            float(metrics[0]),
            float(metrics[1]),
            float(metrics[2]),
            float(metrics[3]),
        )
        embedding_row = arrays["embedding"][index]
        alternate_row = arrays["alternate_embedding"][index]
        embedding = None if np.isnan(embedding_row).all() else embedding_row.astype(np.float32)
        alternate = None if np.isnan(alternate_row).all() else alternate_row.astype(np.float32)
        observation = _Observation(
            frame_index=frame_index,
            second=frame_index / fps,
            detection=FaceDetection(
                bbox=tuple(int(value) for value in arrays["bbox"][index]),
                landmarks=arrays["landmarks"][index].astype(np.float32),
                confidence=float(arrays["detection_confidence"][index]),
            ),
            quality=quality,
            embedding=embedding,
            scene_id=int(arrays["scene_id"][index]),
            alternate_embedding=alternate,
        )
        frame_observations[frame_index].append(observation)
    return frame_observations


def _save_feature_cache(path: Path | str, metadata: dict, arrays: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["metadata_json"] = np.array([json.dumps(metadata, ensure_ascii=False)], dtype=np.str_)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temp_path.replace(path)


def _load_feature_cache(path: Path | str) -> tuple[dict, dict]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        metadata_raw = payload["metadata_json"]
        if metadata_raw.size == 0:
            raise ValueError("La cache de features no tiene metadata.")
        metadata = json.loads(str(metadata_raw[0]))
        if int(metadata.get("version", 0)) != FEATURE_CACHE_VERSION:
            raise ValueError("Version de cache de features incompatible.")
        arrays = {key: payload[key] for key in payload.files if key != "metadata_json"}
    return metadata, arrays


def _flush_pending_frames(
    pending_frames,
    embedder,
    celebrity_index,
    tracks,
    frame_observations,
    config,
    stats,
):
    aligned_items = [item for _, _, items in pending_frames for item in items]
    if aligned_items:
        aligned_faces = [aligned for _, aligned in aligned_items]
        try:
            embeddings = embedder.embed_tta_batch(aligned_faces)
        except Exception:
            embeddings = []
            for observation, aligned in aligned_items:
                try:
                    embeddings.append(_tta_from_aligned(aligned, embedder))
                except Exception:
                    embeddings.append(None)
                    observation.quality.reason = _append_reason(
                        observation.quality.reason,
                        "embedding_error",
                    )
                    stats["embedding_errors"] += 1
        for (observation, _), embedding in zip(aligned_items, embeddings):
            observation.embedding = embedding

    _track_pending_frames(
        pending_frames,
        celebrity_index,
        tracks,
        frame_observations,
        config,
    )


def _complete_pending_frames(
    pending_frames,
    future,
    embedder,
    celebrity_index,
    tracks,
    frame_observations,
    config,
    stats,
):
    aligned_items = [item for _, _, items in pending_frames for item in items]
    try:
        embeddings = future.result()
    except Exception:
        embeddings = []
        for observation, aligned in aligned_items:
            try:
                embeddings.append(_tta_from_aligned(aligned, embedder))
            except Exception:
                embeddings.append(None)
                observation.quality.reason = _append_reason(
                    observation.quality.reason,
                    "embedding_error",
                )
                stats["embedding_errors"] += 1
    for (observation, _), embedding in zip(aligned_items, embeddings):
        observation.embedding = embedding
    _track_pending_frames(
        pending_frames,
        celebrity_index,
        tracks,
        frame_observations,
        config,
    )


def _track_pending_frames(
    pending_frames,
    celebrity_index,
    tracks,
    frame_observations,
    config,
):
    for frame_index, observations, _ in pending_frames:
        for observation in observations:
            if observation.embedding is not None:
                observation.top_matches = celebrity_index.top_unique(observation.embedding, limit=3)
        _assign_tracks(tracks, observations, config)
        frame_observations[frame_index].extend(observations)


def _multiscale_detections(
    frame,
    primary_detector,
    secondary_detector,
    config,
    frame_index=0,
    scene_cut=False,
):
    detections = list(primary_detector.detect(frame))
    if not _needs_secondary_scale(detections, config, frame_index, scene_cut):
        return _nms_detections(detections, config.nms_iou, config.max_faces)
    scale = config.secondary_scale
    scaled = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for detection in secondary_detector.detect(scaled):
        landmarks = detection.landmarks.copy()
        landmarks[:, :2] /= scale
        x1, y1, x2, y2 = detection.bbox
        detections.append(
            FaceDetection(
                bbox=(int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)),
                landmarks=landmarks,
                confidence=detection.confidence,
            )
        )
    return _nms_detections(detections, config.nms_iou, config.max_faces)


def _needs_secondary_scale(detections, config, frame_index: int, scene_cut: bool) -> bool:
    if config.secondary_strategy == "always":
        return True
    if config.secondary_strategy != "adaptive":
        raise ValueError("secondary_strategy debe ser 'always' o 'adaptive'.")
    if scene_cut or not detections or frame_index % max(1, config.secondary_scan_interval) == 0:
        return True
    return any(min(det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]) <= config.secondary_trigger_side for det in detections)


def _nms_detections(detections, threshold: float, limit: int):
    ordered = sorted(detections, key=lambda item: _bbox_area(item.bbox), reverse=True)
    kept = []
    for detection in ordered:
        if any(_bbox_iou(detection.bbox, current.bbox) >= threshold for current in kept):
            continue
        kept.append(detection)
        if len(kept) >= limit:
            break
    return kept


def _measure_quality(frame, detection, config) -> _Quality:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    side = max(0, min(x2 - x1, y2 - y1))
    if side < config.min_face_side:
        return _Quality(False, 0.0, "rostro_muy_chico", side, 0.0, 0.0, 0.0, 0.0)

    left_eye, right_eye = eye_centers(detection.landmarks)
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    if eye_distance < config.min_eye_distance:
        return _Quality(False, 0.0, "resolucion_insuficiente", side, eye_distance, 0.0, 0.0, 0.0)

    roi = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
    if roi.size == 0:
        return _Quality(False, 0.0, "fuera_de_cuadro", side, eye_distance, 0.0, 0.0, 0.0)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    landmarks = detection.landmarks
    eye_axis = right_eye - left_eye
    eye_axis_unit = eye_axis / max(float(np.linalg.norm(eye_axis)), 1e-6)
    eye_center = (left_eye + right_eye) * 0.5
    nose_shift = float(np.dot(landmarks[NOSE_TIP, :2] - eye_center, eye_axis_unit) / eye_distance)
    face_height = float(np.linalg.norm(landmarks[CHIN, :2] - eye_center) / eye_distance)
    mouth_width = float(
        np.linalg.norm(landmarks[RIGHT_MOUTH, :2] - landmarks[LEFT_MOUTH, :2]) / eye_distance
    )

    reasons = []
    if blur < 15.0:
        reasons.append("desenfoque")
    if abs(nose_shift) > 0.32:
        reasons.append("pose")
    if brightness < 30 or brightness > 235:
        reasons.append("iluminacion")
    if face_height < 0.62 or face_height > 2.50 or mouth_width < 0.28:
        reasons.append("geometria")

    severe = (
        brightness < 18
        or brightness > 245
        or abs(nose_shift) > 0.58
        or face_height < 0.42
        or face_height > 3.0
    )
    size_score = float(np.clip((side - config.min_face_side) / 220.0, 0.0, 1.0))
    blur_score = float(np.clip((blur - 2.0) / 80.0, 0.0, 1.0))
    light_score = float(1.0 - np.clip(abs(brightness - 128.0) / 128.0, 0.0, 1.0))
    pose_score = float(1.0 - np.clip(abs(nose_shift) / 0.58, 0.0, 1.0))
    weight = max(0.05, 0.30 * size_score + 0.25 * blur_score + 0.20 * light_score + 0.25 * pose_score)
    return _Quality(
        not severe,
        weight if not severe else 0.0,
        ",".join(reasons) if reasons else "ok",
        side,
        eye_distance,
        brightness,
        blur,
        nose_shift,
    )


def _tta_embedding(frame, detection, embedder):
    embedding, aligned = embedder.embed_face(frame, detection)
    flipped = embedder.embed(cv2.flip(aligned, 1))
    combined = embedding.astype(np.float32) + flipped.astype(np.float32)
    return _normalize(combined)


def _tta_from_aligned(aligned, embedder):
    embedding = embedder.embed(aligned)
    flipped = embedder.embed(cv2.flip(aligned, 1))
    return _normalize(embedding.astype(np.float32) + flipped.astype(np.float32))


def _assign_tracks(tracks, observations, config):
    assigned = set()
    for observation in sorted(observations, key=lambda item: _bbox_area(item.detection.bbox), reverse=True):
        candidates = []
        for track in tracks:
            if track.track_id in assigned:
                continue
            gap = observation.frame_index - track.last.frame_index
            if gap < 1 or gap > config.max_track_gap:
                continue
            if track.last.scene_id != observation.scene_id:
                continue
            score = _track_score(track.last, observation)
            if score >= 0.20:
                candidates.append((score, track))
        if candidates:
            _, track = max(candidates, key=lambda item: item[0])
        else:
            track = _Track(len(tracks))
            tracks.append(track)
        observation.track_id = track.track_id
        track.observations.append(observation)
        assigned.add(track.track_id)


def _track_score(previous, current):
    iou = _bbox_iou(previous.detection.bbox, current.detection.bbox)
    center_score = _bbox_center_score(previous.detection.bbox, current.detection.bbox)
    embedding_score = 0.0
    if previous.embedding is not None and current.embedding is not None:
        embedding_score = max(0.0, float(previous.embedding @ current.embedding))
        if embedding_score < 0.45:
            return 0.0
    spatial_score = max(iou, center_score)
    if spatial_score < 0.10 and embedding_score < 0.72:
        return 0.0
    return 0.52 * spatial_score + 0.48 * embedding_score


def _merge_compatible_tracks(tracks, config):
    tracks = sorted(tracks, key=lambda item: item.observations[0].frame_index)
    changed = True
    while changed:
        changed = False
        for left_index, left in enumerate(tracks):
            for right_index in range(left_index + 1, len(tracks)):
                right = tracks[right_index]
                gap = right.observations[0].frame_index - left.observations[-1].frame_index
                if gap <= 0 or gap > config.max_fill_gap:
                    continue
                if left.observations[-1].scene_id != right.observations[0].scene_id:
                    continue
                spatial = max(
                    _bbox_iou(left.observations[-1].detection.bbox, right.observations[0].detection.bbox),
                    _bbox_center_score(
                        left.observations[-1].detection.bbox,
                        right.observations[0].detection.bbox,
                    ),
                )
                left_embedding = _track_mean_embedding(left)
                right_embedding = _track_mean_embedding(right)
                similarity = (
                    float(left_embedding @ right_embedding)
                    if left_embedding is not None and right_embedding is not None
                    else 0.0
                )
                if not ((spatial >= 0.20 and similarity >= 0.42) or similarity >= 0.78):
                    continue
                for observation in right.observations:
                    observation.track_id = left.track_id
                left.observations.extend(right.observations)
                left.observations.sort(key=lambda item: item.frame_index)
                left.extra_embeddings.extend(right.extra_embeddings)
                tracks.pop(right_index)
                changed = True
                break
            if changed:
                break
    return tracks


def _track_mean_embedding(track):
    embeddings = [item.embedding for item in track.observations if item.embedding is not None]
    if not embeddings:
        return None
    return _normalize(np.mean(np.vstack(embeddings), axis=0))


def _decide_track(track, celebrity_index, config):
    observations = [item for item in track.observations if item.embedding is not None]
    alternate_embeddings = [
        item.alternate_embedding
        for item in track.observations
        if item.alternate_embedding is not None
    ]
    embeddings = [item.embedding for item in observations] + alternate_embeddings + track.extra_embeddings
    if len(embeddings) < config.min_track_embeddings:
        return _TrackDecision(
            evidence_frames=len(observations),
            rejection_reason="evidencia_insuficiente",
        )
    matrix = np.vstack(embeddings).astype(np.float32)
    weights = np.array(
        [item.quality.weight for item in observations]
        + [0.55] * (len(alternate_embeddings) + len(track.extra_embeddings)),
        dtype=np.float32,
    )
    aggregate = _robust_embedding(matrix, weights)
    matches = celebrity_index.top_unique(aggregate, limit=5)
    if celebrity_match_rejection_reason(matches, config.min_similarity, config.min_similarity_margin):
        return _TrackDecision(
            top_matches=matches,
            evidence_frames=len(observations),
            rejection_reason="agregado_ambiguo",
        )

    best = matches[0]
    votes = 0
    competing_votes = 0
    for item in observations:
        if not item.top_matches:
            continue
        accepted = celebrity_match_rejection_reason(
            item.top_matches,
            config.min_similarity,
            config.min_similarity_margin,
        ) is None
        if not accepted:
            continue
        if item.top_matches[0].name == best.name:
            votes += 1
        else:
            competing_votes += 1

    evidence_frames = len(observations)
    vote_ratio = votes / max(1, evidence_frames)
    competing_ratio = competing_votes / max(1, evidence_frames)
    required_votes = max(
        config.min_temporal_votes,
        int(np.ceil(evidence_frames * config.temporal_vote_ratio)),
    )

    primary_matrix = np.vstack([item.embedding for item in observations]).astype(np.float32)
    aggregate_similarities = primary_matrix @ aggregate
    median_similarity = float(np.median(aggregate_similarities))
    mad = float(np.median(np.abs(aggregate_similarities - median_similarity)))
    inlier_floor = max(0.30, median_similarity - 3.5 * max(mad, 0.01))
    inlier_ratio = float(np.mean(aggregate_similarities >= inlier_floor))

    sample_embeddings = celebrity_index.sample_embeddings_for_person(best.name)
    support_threshold = max(0.28, config.min_similarity - config.sample_support_slack)
    support_count = 0
    if sample_embeddings.size:
        sample_similarities = sample_embeddings @ aggregate
        support_count = int(np.count_nonzero(sample_similarities >= support_threshold))
    required_support = min(config.sample_support_count, len(sample_embeddings))
    rejection_reason = ""
    strong_aggregate = best.similarity >= config.min_similarity + 0.08
    strong_track_rescue = (
        strong_aggregate
        and votes >= config.min_temporal_votes
        and vote_ratio >= config.strong_track_min_vote_ratio
        and competing_ratio <= config.max_competing_vote_ratio
    )
    if support_count < required_support:
        rejection_reason = "soporte_individual_insuficiente"
    elif (votes < required_votes or vote_ratio < config.temporal_vote_ratio) and not strong_track_rescue:
        rejection_reason = "votos_temporales_insuficientes"
    elif competing_ratio > config.max_competing_vote_ratio:
        rejection_reason = "identidades_competidoras"
    elif inlier_ratio < config.min_embedding_inlier_ratio:
        rejection_reason = "track_inconsistente"
    if rejection_reason:
        return _TrackDecision(
            top_matches=matches,
            support_count=support_count,
            votes=votes,
            competing_votes=competing_votes,
            evidence_frames=evidence_frames,
            vote_ratio=vote_ratio,
            inlier_ratio=inlier_ratio,
            rejection_reason=rejection_reason,
        )

    return _TrackDecision(
        label=best.name,
        confidence=min(0.99, max(0.0, best.similarity)),
        similarity=best.similarity,
        distance=best.distance,
        top_matches=matches,
        support_count=support_count,
        votes=votes,
        competing_votes=competing_votes,
        evidence_frames=evidence_frames,
        vote_ratio=vote_ratio,
        inlier_ratio=inlier_ratio,
    )


def _robust_embedding(matrix, weights):
    centroid = _normalize(np.average(matrix, axis=0, weights=np.maximum(weights, 0.05)))
    similarities = matrix @ centroid
    cutoff = max(float(np.quantile(similarities, 0.20)), 0.35)
    keep = similarities >= cutoff
    if np.count_nonzero(keep) < min(3, len(matrix)):
        keep = np.argsort(similarities)[-min(3, len(matrix)) :]
        selected = matrix[keep]
        selected_weights = weights[keep]
    else:
        selected = matrix[keep]
        selected_weights = weights[keep]
    return _normalize(np.average(selected, axis=0, weights=np.maximum(selected_weights, 0.05)))


def _frame_decisions_for_tracks(tracks, config):
    """Resuelve conocido/desconocido usando todo el track sin fijar su confianza."""
    decisions = {}
    for track in tracks:
        observations = sorted(track.observations, key=lambda item: item.frame_index)
        if not observations:
            continue
        if track.decision.label == "desconocido":
            for observation in observations:
                decisions[id(observation)] = _FrameDecision(
                    reason=track.decision.rejection_reason or "track_desconocido"
                )
            continue

        emissions = []
        local_rows = []
        for observation in observations:
            row = _local_identity_evidence(observation, track.decision.label, config)
            local_rows.append(row)
            emissions.append(row["emission"])

        temporal_probabilities = _bidirectional_known_probabilities(
            np.asarray(emissions, dtype=np.float64),
            config,
        )
        competing_vetoes = _confirmed_competing_segments(
            local_rows,
            temporal_probabilities,
            config,
        )
        track_confidence = _track_display_confidence(track.decision, config)
        for observation, row, temporal_probability, competing_veto in zip(
            observations,
            local_rows,
            temporal_probabilities,
            competing_vetoes,
        ):
            is_known = not competing_veto
            temporal_confidence = max(
                float(temporal_probability), config.accepted_track_probability_floor
            )
            frame_confidence = (
                0.85 * row["local_confidence"]
                + 0.15 * temporal_confidence
            )
            confidence = float(
                np.clip(
                    config.global_confidence_weight * track_confidence
                    + config.frame_confidence_weight * frame_confidence,
                    0.0,
                    0.99,
                )
            )
            decisions[id(observation)] = _FrameDecision(
                label=track.decision.label if is_known else "desconocido",
                confidence=confidence if is_known else 0.0,
                distance=row["distance"] if is_known else float("inf"),
                similarity=row["similarity"],
                margin=row["margin"],
                local_confidence=row["local_confidence"],
                temporal_probability=float(temporal_probability),
                reason=row["reason"] if is_known else "identidad_competidora_sostenida",
            )
    return decisions


def _local_identity_evidence(observation, label, config):
    if observation.embedding is None or not observation.top_matches:
        return {
            "similarity": 0.0,
            "margin": 0.0,
            "distance": float("inf"),
            "local_confidence": 0.0,
            "emission": 0.50,
            "reason": "interpolado" if observation.synthetic else "sin_embedding",
        }

    candidate = next((item for item in observation.top_matches if item.name == label), None)
    best = observation.top_matches[0]
    competing = max(
        (item.similarity for item in observation.top_matches if item.name != label),
        default=-1.0,
    )
    similarity = float(candidate.similarity) if candidate is not None else 0.0
    margin = similarity - float(competing)
    similarity_score = float(
        np.clip(
            (similarity - config.min_similarity) / max(config.confidence_similarity_span, 1e-6),
            0.0,
            1.0,
        )
    )
    margin_score = float(
        np.clip(
            (margin - config.min_similarity_margin) / max(config.confidence_margin_span, 1e-6),
            0.0,
            1.0,
        )
    )
    quality_score = float(np.clip(observation.quality.weight, 0.0, 1.0))
    similarity_confidence = float(np.clip(0.35 + 0.65 * similarity_score, 0.0, 1.0))
    margin_confidence = float(
        np.clip(0.35 + 0.65 * margin_score, 0.0, 1.0)
    )
    local_confidence = (
        0.60 * similarity_confidence
        + 0.25 * margin_confidence
        + 0.15 * quality_score
    )

    best_is_competitor = best.name != label and celebrity_match_rejection_reason(
        observation.top_matches,
        config.min_similarity,
        config.min_similarity_margin,
    ) is None
    if best_is_competitor:
        local_confidence = 0.20 * quality_score
        emission = 0.03
        reason = "identidad_competidora"
    elif candidate is None:
        local_confidence = 0.15 * quality_score
        emission = 0.38
        reason = "candidato_ausente"
    elif similarity < config.min_similarity:
        emission = 0.38 + 0.12 * similarity_score
        reason = "similitud_baja"
    elif margin < config.min_similarity_margin:
        emission = 0.48 + 0.10 * similarity_score
        reason = "margen_bajo"
    else:
        emission = 0.62 + 0.36 * local_confidence
        reason = "evidencia_local"

    return {
        "similarity": similarity,
        "margin": margin,
        "distance": float(candidate.distance) if candidate is not None else float("inf"),
        "local_confidence": float(local_confidence),
        "emission": float(np.clip(emission, 0.01, 0.99)),
        "reason": reason,
    }


def _track_display_confidence(decision, config):
    """Puntaje estable del track; no pretende ser una probabilidad estadistica."""
    similarity_component = 0.55 + 0.43 * float(
        np.clip(
            (decision.similarity - config.min_similarity)
            / max(config.confidence_similarity_span, 1e-6),
            0.0,
            1.0,
        )
    )
    if len(decision.top_matches) > 1:
        aggregate_margin = decision.top_matches[0].similarity - decision.top_matches[1].similarity
    else:
        aggregate_margin = config.min_similarity_margin + config.confidence_margin_span
    margin_component = 0.50 + 0.50 * float(
        np.clip(
            (aggregate_margin - config.min_similarity_margin)
            / max(config.confidence_margin_span, 1e-6),
            0.0,
            1.0,
        )
    )
    support_component = float(
        np.clip(
            decision.support_count / max(1, config.sample_support_count),
            0.0,
            1.0,
        )
    )
    return float(
        np.clip(
            0.40 * similarity_component
            + 0.25 * decision.vote_ratio
            + 0.15 * decision.inlier_ratio
            + 0.10 * support_component
            + 0.10 * margin_component,
            0.50,
            0.99,
        )
    )


def _bidirectional_known_probabilities(known_emissions, config):
    if known_emissions.size == 0:
        return np.empty(0, dtype=np.float64)
    emissions = np.column_stack((1.0 - known_emissions, known_emissions))
    transition = np.array(
        [
            [config.unknown_state_persistence, 1.0 - config.unknown_state_persistence],
            [1.0 - config.known_state_persistence, config.known_state_persistence],
        ],
        dtype=np.float64,
    )
    forward = np.empty_like(emissions)
    forward[0] = np.array([0.05, 0.95]) * emissions[0]
    forward[0] /= max(float(forward[0].sum()), 1e-12)
    for index in range(1, len(emissions)):
        forward[index] = (forward[index - 1] @ transition) * emissions[index]
        forward[index] /= max(float(forward[index].sum()), 1e-12)

    backward = np.ones_like(emissions)
    for index in range(len(emissions) - 2, -1, -1):
        backward[index] = transition @ (emissions[index + 1] * backward[index + 1])
        backward[index] /= max(float(backward[index].sum()), 1e-12)

    posterior = forward * backward
    posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-12)
    return posterior[:, 1]


def _confirmed_competing_segments(local_rows, temporal_probabilities, config):
    """Veta solo contradicciones largas dominadas por otra identidad conocida."""
    vetoes = np.zeros(len(local_rows), dtype=np.bool_)
    low_probability = temporal_probabilities < config.frame_known_probability
    index = 0
    while index < len(local_rows):
        if not low_probability[index]:
            index += 1
            continue
        end = index + 1
        while end < len(local_rows) and low_probability[end]:
            end += 1
        segment = local_rows[index:end]
        competing = sum(row["reason"] == "identidad_competidora" for row in segment)
        if (
            len(segment) >= config.min_competing_segment_frames
            and competing / len(segment) >= config.competing_segment_ratio
        ):
            vetoes[index:end] = True
        index = end
    return vetoes


def _refine_unresolved_tracks(
    video_path,
    tracks,
    embedder,
    celebrity_index,
    config,
    cancel_event,
    progress_callback,
):
    if not tracks or getattr(embedder, "backend_name", "") != "deepface":
        return
    tracks = [track for track in tracks if _needs_track_refinement(track, config)]
    if not tracks:
        return
    cap = cv2.VideoCapture(str(video_path))
    candidates = []
    try:
        for index, track in enumerate(tracks):
            _check_cancel(cancel_event)
            usable = [
                item
                for item in track.observations
                if item.quality.usable and item.alternate_embedding is None
            ]
            selected = sorted(usable, key=lambda item: item.quality.weight, reverse=True)[
                : config.max_refinement_frames
            ]
            for observation in selected:
                cap.set(cv2.CAP_PROP_POS_FRAMES, observation.frame_index)
                ok, frame = cap.read()
                if not ok:
                    continue
                try:
                    crop = _padded_crop(frame, observation.detection.bbox, padding=0.30)
                    aligned = embedder._deepface_aligned_face(crop)
                    candidates.append((track, observation, aligned))
                except Exception:
                    continue
            if progress_callback and (index % 8 == 0 or index + 1 == len(tracks)):
                progress_callback(
                    0.84 + 0.12 * (index + 1) / len(tracks),
                    f"Segunda pasada: track {index + 1}/{len(tracks)}",
                )
    finally:
        cap.release()

    batch_size = max(1, config.embedding_batch_size)
    for offset in range(0, len(candidates), batch_size):
        chunk = candidates[offset : offset + batch_size]
        try:
            alternates = embedder.embed_batch([item[2] for item in chunk])
        except Exception:
            alternates = []
            for _, _, aligned in chunk:
                try:
                    alternates.append(embedder.embed(aligned))
                except Exception:
                    alternates.append(None)
        for (_, observation, _), alternate in zip(chunk, alternates):
            if alternate is not None and (
                observation.embedding is None
                or float(alternate @ observation.embedding) >= 0.20
            ):
                observation.alternate_embedding = alternate
    for track in tracks:
        if any(item.alternate_embedding is not None for item in track.observations):
            track.decision = _decide_track(track, celebrity_index, config)


def _needs_track_refinement(track, config) -> bool:
    embeddings = sum(item.embedding is not None for item in track.observations)
    matches = track.decision.top_matches
    if embeddings < max(config.min_track_embeddings, 8):
        return True
    if not matches:
        return True
    return matches[0].similarity >= config.min_similarity - 0.10


def _fill_short_track_gaps(frame_observations, tracks, config, fps):
    count = 0
    for track in tracks:
        if track.decision.label == "desconocido":
            continue
        observations = sorted(track.observations, key=lambda item: item.frame_index)
        for previous, current in zip(observations, observations[1:]):
            gap = current.frame_index - previous.frame_index - 1
            if gap <= 0 or gap > config.max_fill_gap:
                continue
            if _bbox_iou(previous.detection.bbox, current.detection.bbox) < 0.12:
                continue
            for offset in range(1, gap + 1):
                frame_index = previous.frame_index + offset
                alpha = offset / (gap + 1)
                detection = _interpolate_detection(previous.detection, current.detection, alpha)
                if any(_bbox_iou(detection.bbox, item.detection.bbox) > 0.65 for item in frame_observations[frame_index]):
                    continue
                quality = _Quality(True, 0.0, "interpolado", 0, 0.0, 0.0, 0.0, 0.0)
                synthetic = _Observation(
                    frame_index=frame_index,
                    second=frame_index / fps,
                    detection=detection,
                    quality=quality,
                    embedding=None,
                    track_id=track.track_id,
                    synthetic=True,
                    scene_id=previous.scene_id,
                )
                frame_observations[frame_index].append(synthetic)
                track.observations.append(synthetic)
                count += 1
        track.observations.sort(key=lambda item: item.frame_index)
    return count


def _records_from_observations(frame_observations, tracks, fps, config):
    decisions = {track.track_id: track.decision for track in tracks}
    frame_decisions = _frame_decisions_for_tracks(tracks, config)
    records = []
    for frame_index, observations in enumerate(frame_observations):
        faces = []
        for observation in observations:
            decision = decisions[observation.track_id]
            frame_decision = frame_decisions[id(observation)]
            raw_matches = observation.top_matches[:3]
            faces.append(
                {
                    "bbox": [int(value) for value in observation.detection.bbox],
                    "landmarks": [
                        [int(round(point[0])), int(round(point[1])), round(float(point[2]), 6)]
                        for point in observation.detection.landmarks
                    ],
                    "detection_confidence": float(observation.detection.confidence),
                    "label": frame_decision.label,
                    "confidence": frame_decision.confidence,
                    "distance": frame_decision.distance
                    if np.isfinite(frame_decision.distance)
                    else None,
                    "method": "offline_bidireccional"
                    if frame_decision.label != "desconocido"
                    else "offline_rechazo",
                    "track_id": observation.track_id,
                    "synthetic": observation.synthetic,
                    "quality": observation.quality.as_dict(),
                    "raw_matches": [
                        {"name": item.name, "similarity": round(item.similarity, 6)}
                        for item in raw_matches
                    ],
                    "track_evidence": {
                        "similarity": round(decision.similarity, 6),
                        "support_count": decision.support_count,
                        "votes": decision.votes,
                        "competing_votes": decision.competing_votes,
                        "evidence_frames": decision.evidence_frames,
                        "vote_ratio": round(decision.vote_ratio, 6),
                        "inlier_ratio": round(decision.inlier_ratio, 6),
                        "rejection_reason": decision.rejection_reason,
                        "display_confidence": round(
                            _track_display_confidence(decision, config),
                            6,
                        )
                        if decision.label != "desconocido"
                        else 0.0,
                    },
                    "frame_evidence": {
                        "similarity": round(frame_decision.similarity, 6),
                        "margin": round(frame_decision.margin, 6),
                        "local_confidence": round(frame_decision.local_confidence, 6),
                        "temporal_probability": round(
                            frame_decision.temporal_probability,
                            6,
                        ),
                        "reason": frame_decision.reason,
                    },
                }
            )
        records.append({"seconds": frame_index / fps, "faces": faces})
    return records


def _interpolate_detection(a, b, alpha):
    bbox = tuple(int(round((1 - alpha) * x + alpha * y)) for x, y in zip(a.bbox, b.bbox))
    landmarks = (1 - alpha) * a.landmarks + alpha * b.landmarks
    return FaceDetection(bbox=bbox, landmarks=landmarks.astype(np.float32), confidence=min(a.confidence, b.confidence))


def _padded_crop(frame, bbox, padding=0.30):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    dx, dy = int((x2 - x1) * padding), int((y2 - y1) * padding)
    return frame[max(0, y1 - dy) : min(h, y2 + dy), max(0, x1 - dx) : min(w, x2 + dx)].copy()


def _append_reason(reason, extra):
    return extra if reason == "ok" else f"{reason},{extra}"


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return float(intersection / union) if union else 0.0


def _bbox_center_score(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    distance = float(np.hypot((ax1 + ax2 - bx1 - bx2) / 2, (ay1 + ay2 - by1 - by2) / 2))
    scale = max(np.sqrt(max(_bbox_area(a), _bbox_area(b))), 1.0)
    return max(0.0, 1.0 - distance / (1.8 * scale))


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Analisis cancelado.")


def _scene_signature(frame):
    small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram)
    return gray, histogram


def _is_scene_cut(previous, current):
    previous_gray, previous_histogram = previous
    current_gray, current_histogram = current
    pixel_change = float(np.mean(cv2.absdiff(previous_gray, current_gray))) / 255.0
    histogram_correlation = float(
        cv2.compareHist(previous_histogram, current_histogram, cv2.HISTCMP_CORREL)
    )
    return pixel_change >= 0.16 and histogram_correlation <= 0.45

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import List, Optional, Tuple
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from .logging_config import configure_absl_logs


configure_absl_logs()


@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]
    landmarks: np.ndarray
    confidence: float


MODEL_CACHE_DIR = Path(tempfile.gettempdir()) / "tp_integrador_mediapipe"
FACE_LANDMARKER_MODEL_PATH = MODEL_CACHE_DIR / "face_landmarker.task"
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MIN_MODEL_SIZE_BYTES = 1024 * 1024


class MediaPipeFaceDetector:
    """Deteccion y landmarks faciales sin cascadas ni HOG."""

    def __init__(
        self,
        max_faces: int = 4,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._backend = "solutions"
        self._timestamp_ms = 0

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        self._backend = "tasks"
        model_path = self._ensure_face_landmarker_model()
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._face_mesh = FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._face_mesh.close()

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._backend == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            self._timestamp_ms += 33
            result = self._face_mesh.detect_for_video(mp_image, self._timestamp_ms)
            return self._detections_from_landmarks(result.face_landmarks, w, h)

        result = self._face_mesh.process(frame_rgb)
        return self._detections_from_landmarks(result.multi_face_landmarks, w, h)

    def _detections_from_landmarks(self, faces, w: int, h: int) -> List[FaceDetection]:
        detections: List[FaceDetection] = []
        if not faces:
            return detections

        for face_landmarks in faces:
            landmarks = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
            points = np.array(
                [(lm.x * w, lm.y * h, lm.z) for lm in landmarks],
                dtype=np.float32,
            )
            x1, y1 = np.min(points[:, :2], axis=0)
            x2, y2 = np.max(points[:, :2], axis=0)
            pad_x = 0.08 * (x2 - x1)
            pad_y = 0.12 * (y2 - y1)
            x1 = int(max(0, x1 - pad_x))
            y1 = int(max(0, y1 - pad_y))
            x2 = int(min(w - 1, x2 + pad_x))
            y2 = int(min(h - 1, y2 + pad_y))
            detections.append(FaceDetection((x1, y1, x2, y2), points, 1.0))

        return detections

    def _ensure_face_landmarker_model(self) -> Path:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if _valid_model_file(FACE_LANDMARKER_MODEL_PATH):
            return FACE_LANDMARKER_MODEL_PATH

        print(f"Descargando modelo MediaPipe en ruta corta: {FACE_LANDMARKER_MODEL_PATH}")
        tmp_path = FACE_LANDMARKER_MODEL_PATH.with_suffix(".download")
        try:
            urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, tmp_path)
            tmp_path.replace(FACE_LANDMARKER_MODEL_PATH)
        except Exception as exc:
            raise FileNotFoundError(
                "No se pudo descargar face_landmarker.task. Descargalo manualmente desde "
                f"{FACE_LANDMARKER_MODEL_URL} y guardalo como {FACE_LANDMARKER_MODEL_PATH}"
            ) from exc

        if not _valid_model_file(FACE_LANDMARKER_MODEL_PATH):
            raise FileNotFoundError(
                f"El modelo descargado no parece valido: {FACE_LANDMARKER_MODEL_PATH}"
            )
        return FACE_LANDMARKER_MODEL_PATH


def _valid_model_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_MODEL_SIZE_BYTES


def largest_face(detections: List[FaceDetection]) -> Optional[FaceDetection]:
    if not detections:
        return None
    return max(detections, key=lambda det: (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]))

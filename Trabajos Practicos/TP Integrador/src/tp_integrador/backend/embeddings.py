import os

import cv2
import numpy as np

from .alineamiento import align_face
from .deteccion import FaceDetection


class ArcFaceEmbedder:
    """Extrae embeddings ArcFace de 512 dimensiones.

    Backends disponibles:
    - auto: usa DeepFace con ArcFace, que es el backend mas robusto en Python moderno.
    - arcface: biblioteca `arcface` del material de clase.
    - deepface: DeepFace con model_name="ArcFace".

    Se elige con TP_FACE_EMBEDDER=auto|arcface|deepface. Por defecto usa deepface.
    """

    def __init__(self) -> None:
        requested_backend = os.environ.get("TP_FACE_EMBEDDER", "deepface").strip().lower()
        if requested_backend not in {"auto", "arcface", "deepface"}:
            raise ValueError("TP_FACE_EMBEDDER debe ser 'auto', 'arcface' o 'deepface'.")

        if requested_backend == "deepface":
            self._init_deepface()
            return

        if requested_backend == "arcface":
            self._init_arcface()
            return

        self._init_deepface()

    def _init_arcface(self) -> None:
        try:
            from arcface import ArcFace

            self._face_rec = ArcFace.ArcFace()
            self._backend = "arcface"
        except Exception as exc:
            raise RuntimeError(
                "No se pudo cargar la libreria arcface o su modelo preentrenado. "
                "Puede faltar astropy o puede estar caida la URL del modelo de PyPI."
            ) from exc

    def _init_deepface(self) -> None:
        # DeepFace sobre TensorFlow 2.16+ necesita tf-keras para compatibilidad con Keras 3.
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        try:
            from deepface import DeepFace

            self._deepface = DeepFace
            self._backend = "deepface"
        except Exception as exc:
            raise RuntimeError(
                "No se pudo cargar DeepFace. En Python 3.12 ejecuta: pip install -r requirements.txt. "
                "Si el error menciona tf_keras, ejecuta: pip install tf-keras"
            ) from exc

    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """Extrae embedding desde una cara ya preprocesada.

        Para DeepFace se mantiene como camino de compatibilidad y no realinea.
        El pipeline principal debe usar `embed_face`.
        """
        if self._backend == "deepface":
            result = self._deepface.represent(
                img_path=aligned_face_bgr,
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False,
                align=False,
            )
            embedding = np.array(result[0]["embedding"], dtype=np.float32)
            return _normalize(embedding)

        embedding = np.array(self._face_rec.calc_emb(aligned_face_bgr), dtype=np.float32)
        return _normalize(embedding)

    def embed_face(self, frame_bgr: np.ndarray, detection: FaceDetection) -> tuple[np.ndarray, np.ndarray]:
        """Extrae embedding desde el frame original y la deteccion de MediaPipe.

        DeepFace hace su propia alineacion. La alineacion manual queda solo para
        el backend `arcface`, que no incluye deteccion ni alineamiento.
        """
        if self._backend == "deepface":
            face_bgr = _crop_detection(frame_bgr, detection)
            aligned_bgr = self._deepface_aligned_face(face_bgr)
            return self.embed(aligned_bgr), aligned_bgr

        aligned_bgr = align_face(frame_bgr, detection)
        return self.embed(aligned_bgr), aligned_bgr

    def _deepface_aligned_face(self, face_bgr: np.ndarray) -> np.ndarray:
        detector_backend = os.environ.get("TP_DEEPFACE_DETECTOR", "opencv").strip().lower()
        try:
            faces = self._deepface.extract_faces(
                img_path=face_bgr,
                detector_backend=detector_backend,
                enforce_detection=False,
                align=True,
                grayscale=False,
            )
            if faces:
                return _deepface_face_to_bgr(faces[0]["face"])
        except Exception:
            pass
        return face_bgr


def _normalize(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype(np.float32)


def _crop_detection(frame_bgr: np.ndarray, detection: FaceDetection, padding: float = 0.22) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    pad_x = int(face_w * padding)
    pad_y = int(face_h * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return frame_bgr[y1:y2, x1:x2].copy()


def _deepface_face_to_bgr(face) -> np.ndarray:
    face_rgb = np.array(face)
    if face_rgb.dtype != np.uint8:
        max_value = float(np.max(face_rgb)) if face_rgb.size else 1.0
        if max_value <= 1.0:
            face_rgb = face_rgb * 255.0
        face_rgb = np.clip(face_rgb, 0, 255).astype(np.uint8)
    if face_rgb.ndim == 2:
        return cv2.cvtColor(face_rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

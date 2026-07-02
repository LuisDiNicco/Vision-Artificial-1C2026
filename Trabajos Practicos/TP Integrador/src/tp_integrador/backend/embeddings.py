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

        self._alignment_backend = os.environ.get("TP_FACE_ALIGNMENT", "mediapipe").strip().lower()
        if self._alignment_backend not in {"mediapipe", "deepface"}:
            raise ValueError("TP_FACE_ALIGNMENT debe ser 'mediapipe' o 'deepface'.")

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

    def embed_batch(self, aligned_faces_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Extrae varios embeddings en una inferencia, conservando el preprocesado actual."""
        if not aligned_faces_bgr:
            return []
        if self._backend != "deepface" or len(aligned_faces_bgr) == 1:
            return [self.embed(face) for face in aligned_faces_bgr]

        results = self._deepface.represent(
            img_path=aligned_faces_bgr,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        return [
            _normalize(np.asarray(per_image[0]["embedding"], dtype=np.float32))
            for per_image in results
        ]

    def embed_tta_batch(self, aligned_faces_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Calcula original + flip en un unico batch y devuelve el mismo TTA normalizado."""
        if not aligned_faces_bgr:
            return []
        inputs = list(aligned_faces_bgr)
        inputs.extend(cv2.flip(face, 1) for face in aligned_faces_bgr)
        embeddings = self.embed_batch(inputs)
        offset = len(aligned_faces_bgr)
        return [
            _normalize(embeddings[index] + embeddings[index + offset])
            for index in range(offset)
        ]

    def align_face_input(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> np.ndarray:
        """Produce exactamente la entrada alineada usada por ``embed_face``."""
        if self._backend == "deepface":
            if self._alignment_backend == "mediapipe":
                try:
                    return align_face(frame_bgr, detection)
                except Exception:
                    pass
            return self._deepface_aligned_face(_crop_detection(frame_bgr, detection))
        return align_face(frame_bgr, detection)

    def embed_face(self, frame_bgr: np.ndarray, detection: FaceDetection) -> tuple[np.ndarray, np.ndarray]:
        """Extrae embedding desde el frame original y la deteccion de MediaPipe.

        Por defecto se usa la alineacion de 5 puntos de MediaPipe para darle a
        ArcFace una entrada canonica y estable. DeepFace sigue siendo el
        extractor de embeddings cuando el backend elegido es `deepface`.
        """
        if self._backend == "deepface":
            if self._alignment_backend == "mediapipe":
                try:
                    aligned_bgr = align_face(frame_bgr, detection)
                    return self.embed(aligned_bgr), aligned_bgr
                except Exception:
                    pass
            face_bgr = _crop_detection(frame_bgr, detection)
            aligned_bgr = self._deepface_aligned_face(face_bgr)
            return self.embed(aligned_bgr), aligned_bgr

        aligned_bgr = align_face(frame_bgr, detection)
        return self.embed(aligned_bgr), aligned_bgr

    @property
    def backend_name(self) -> str:
        return self._backend

    @property
    def alignment_backend(self) -> str:
        return self._alignment_backend

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

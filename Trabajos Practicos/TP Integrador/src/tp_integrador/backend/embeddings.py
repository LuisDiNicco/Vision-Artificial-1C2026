import os

import numpy as np


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


def _normalize(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype(np.float32)

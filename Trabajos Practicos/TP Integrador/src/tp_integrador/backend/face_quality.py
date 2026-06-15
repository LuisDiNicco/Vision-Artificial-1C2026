from dataclasses import dataclass

import cv2
import numpy as np

from .alineamiento import eye_centers
from .deteccion import FaceDetection


@dataclass
class FaceQuality:
    ok: bool
    score: float
    reason: str = ""


def assess_face_quality(frame_bgr: np.ndarray, detection: FaceDetection) -> FaceQuality:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    face_w = max(0, x2 - x1)
    face_h = max(0, y2 - y1)
    min_side = min(face_w, face_h)
    if min_side < 96:
        return FaceQuality(False, 0.0, "Rostro muy chico; acercate un poco.")

    left_eye, right_eye = eye_centers(detection.landmarks)
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    if eye_distance < 34:
        return FaceQuality(False, 0.0, "Ojos muy juntos en pixeles; falta resolucion.")

    roi = frame_bgr[max(0, y1): min(h, y2), max(0, x1): min(w, x2)]
    if roi.size == 0:
        return FaceQuality(False, 0.0, "Rostro fuera de cuadro.")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < 45:
        return FaceQuality(False, 0.0, "Imagen oscura; mejora la luz frontal.")
    if brightness > 220:
        return FaceQuality(False, 0.0, "Imagen sobreexpuesta; baja la luz.")

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < 45:
        return FaceQuality(False, 0.0, "Imagen borrosa; mantenete quieto.")

    margin = min(x1, y1, w - x2, h - y2)
    if margin < 4:
        return FaceQuality(False, 0.0, "Rostro cortado por el borde.")

    size_score = np.clip((min_side - 96) / 180, 0.0, 1.0)
    blur_score = np.clip((blur - 45) / 180, 0.0, 1.0)
    light_score = 1.0 - np.clip(abs(brightness - 128) / 128, 0.0, 1.0)
    score = float(0.45 * size_score + 0.35 * blur_score + 0.20 * light_score)
    return FaceQuality(True, score, "")

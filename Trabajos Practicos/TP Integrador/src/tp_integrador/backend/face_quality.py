from dataclasses import dataclass

import cv2
import numpy as np

from .alineamiento import eye_centers
from .deteccion import FaceDetection


NOSE_TIP = 1
CHIN = 152
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


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

    if len(detection.landmarks) > CHIN:
        pose_ok, pose_reason, pose_score = _assess_frontal_pose(detection, left_eye, right_eye, eye_distance)
        if not pose_ok:
            return FaceQuality(False, 0.0, pose_reason)
    else:
        pose_score = 0.75

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
    score = float(0.35 * size_score + 0.25 * blur_score + 0.20 * light_score + 0.20 * pose_score)
    return FaceQuality(True, score, "")


def _assess_frontal_pose(
    detection: FaceDetection,
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    eye_distance: float,
) -> tuple[bool, str, float]:
    landmarks = detection.landmarks
    nose = landmarks[NOSE_TIP, :2]
    chin = landmarks[CHIN, :2]
    left_mouth = landmarks[LEFT_MOUTH, :2]
    right_mouth = landmarks[RIGHT_MOUTH, :2]

    eye_axis = right_eye - left_eye
    eye_axis_unit = eye_axis / max(float(np.linalg.norm(eye_axis)), 1e-6)
    eye_center = (left_eye + right_eye) * 0.5
    nose_shift = float(np.dot(nose - eye_center, eye_axis_unit) / max(eye_distance, 1e-6))
    if abs(nose_shift) > 0.22:
        return False, "Rostro muy girado; mira mas de frente.", 0.0

    mouth_width = float(np.linalg.norm(right_mouth - left_mouth) / max(eye_distance, 1e-6))
    if mouth_width < 0.35:
        return False, "Boca poco visible; evita perfil extremo.", 0.0

    face_height = float(np.linalg.norm(chin - eye_center) / max(eye_distance, 1e-6))
    if face_height < 0.75 or face_height > 2.25:
        return False, "Angulo vertical poco confiable; centra mejor la cara.", 0.0

    nose_score = 1.0 - np.clip(abs(nose_shift) / 0.22, 0.0, 1.0)
    mouth_score = np.clip((mouth_width - 0.35) / 0.35, 0.0, 1.0)
    return True, "", float(0.7 * nose_score + 0.3 * mouth_score)

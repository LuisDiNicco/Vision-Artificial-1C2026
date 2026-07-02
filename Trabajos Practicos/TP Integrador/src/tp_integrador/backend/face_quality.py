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


@dataclass(frozen=True)
class FaceQualityConfig:
    min_side: int = 96
    min_eye_distance: float = 34.0
    min_brightness: float = 45.0
    max_brightness: float = 220.0
    min_blur: float = 45.0
    min_margin: int = 4
    # Dos esquinas por ojo no estiman yaw con suficiente precision para usar
    # 0.22 como rechazo duro: rostros visualmente frontales suelen oscilar por
    # encima de ese valor entre frames.
    max_nose_shift: float = 0.38
    min_mouth_width: float = 0.35
    min_face_height: float = 0.75
    max_face_height: float = 2.25


# Los videos suelen tener compresion, motion blur y planos menos frontales que
# una captura de registro. Este perfil solo decide si vale la pena extraer el
# embedding; la similitud facial sigue siendo quien acepta o rechaza al famoso.
VIDEO_FACE_QUALITY = FaceQualityConfig(
    min_side=72,
    min_eye_distance=24.0,
    min_brightness=30.0,
    max_brightness=235.0,
    min_blur=15.0,
    min_margin=1,
    max_nose_shift=0.32,
    min_mouth_width=0.28,
    min_face_height=0.62,
    max_face_height=2.50,
)

# En reconocimiento conviene tolerar movimientos leves entre frames. El
# registro conserva el perfil estricto por defecto para guardar muestras buenas.
WEBCAM_RECOGNITION_FACE_QUALITY = FaceQualityConfig(max_nose_shift=0.48)


def assess_face_quality(
    frame_bgr: np.ndarray,
    detection: FaceDetection,
    config: FaceQualityConfig | None = None,
) -> FaceQuality:
    config = config or FaceQualityConfig()
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    face_w = max(0, x2 - x1)
    face_h = max(0, y2 - y1)
    min_side = min(face_w, face_h)
    if min_side < config.min_side:
        return FaceQuality(False, 0.0, "Rostro muy chico; acercate un poco.")

    left_eye, right_eye = eye_centers(detection.landmarks)
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    if eye_distance < config.min_eye_distance:
        return FaceQuality(False, 0.0, "Ojos muy juntos en pixeles; falta resolucion.")

    if len(detection.landmarks) > CHIN:
        pose_ok, pose_reason, pose_score = _assess_frontal_pose(
            detection,
            left_eye,
            right_eye,
            eye_distance,
            config,
        )
        if not pose_ok:
            return FaceQuality(False, 0.0, pose_reason)
    else:
        pose_score = 0.75

    roi = frame_bgr[max(0, y1): min(h, y2), max(0, x1): min(w, x2)]
    if roi.size == 0:
        return FaceQuality(False, 0.0, "Rostro fuera de cuadro.")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < config.min_brightness:
        return FaceQuality(False, 0.0, "Imagen oscura; mejora la luz frontal.")
    if brightness > config.max_brightness:
        return FaceQuality(False, 0.0, "Imagen sobreexpuesta; baja la luz.")

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < config.min_blur:
        return FaceQuality(False, 0.0, "Imagen borrosa; mantenete quieto.")

    margin = min(x1, y1, w - x2, h - y2)
    if margin < config.min_margin:
        return FaceQuality(False, 0.0, "Rostro cortado por el borde.")

    size_score = np.clip((min_side - config.min_side) / 180, 0.0, 1.0)
    blur_score = np.clip((blur - config.min_blur) / 180, 0.0, 1.0)
    light_score = 1.0 - np.clip(abs(brightness - 128) / 128, 0.0, 1.0)
    score = float(0.35 * size_score + 0.25 * blur_score + 0.20 * light_score + 0.20 * pose_score)
    return FaceQuality(True, score, "")


def _assess_frontal_pose(
    detection: FaceDetection,
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    eye_distance: float,
    config: FaceQualityConfig,
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
    if abs(nose_shift) > config.max_nose_shift:
        return False, "Rostro muy girado; mira mas de frente.", 0.0

    mouth_width = float(np.linalg.norm(right_mouth - left_mouth) / max(eye_distance, 1e-6))
    if mouth_width < config.min_mouth_width:
        return False, "Boca poco visible; evita perfil extremo.", 0.0

    face_height = float(np.linalg.norm(chin - eye_center) / max(eye_distance, 1e-6))
    if face_height < config.min_face_height or face_height > config.max_face_height:
        return False, "Angulo vertical poco confiable; centra mejor la cara.", 0.0

    nose_score = 1.0 - np.clip(abs(nose_shift) / config.max_nose_shift, 0.0, 1.0)
    mouth_score = np.clip(
        (mouth_width - config.min_mouth_width) / max(config.min_mouth_width, 1e-6),
        0.0,
        1.0,
    )
    return True, "", float(0.7 * nose_score + 0.3 * mouth_score)

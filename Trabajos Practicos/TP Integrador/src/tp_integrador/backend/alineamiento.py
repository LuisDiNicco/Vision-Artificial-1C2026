from typing import Tuple

import cv2
import numpy as np

from .deteccion import FaceDetection


# Indices estables de MediaPipe Face Mesh.
LEFT_EYE = (33, 133)
RIGHT_EYE = (362, 263)


def eye_centers(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    left_eye = np.mean(landmarks[list(LEFT_EYE), :2], axis=0)
    right_eye = np.mean(landmarks[list(RIGHT_EYE), :2], axis=0)
    return left_eye, right_eye


def align_face(
    frame_bgr: np.ndarray,
    detection: FaceDetection,
    output_size: int = 112,
    desired_left_eye: Tuple[float, float] = (0.35, 0.35),
) -> np.ndarray:
    """Alinea el rostro usando los ojos detectados por MediaPipe."""
    left_eye, right_eye = eye_centers(detection.landmarks)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    desired_right_eye_x = 1.0 - desired_left_eye[0]
    desired_dist = (desired_right_eye_x - desired_left_eye[0]) * output_size
    current_dist = np.sqrt((dx * dx) + (dy * dy))
    scale = desired_dist / max(current_dist, 1e-6)

    eyes_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    tx = output_size * 0.5
    ty = output_size * desired_left_eye[1]
    matrix[0, 2] += tx - eyes_center[0]
    matrix[1, 2] += ty - eyes_center[1]

    return cv2.warpAffine(
        frame_bgr,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

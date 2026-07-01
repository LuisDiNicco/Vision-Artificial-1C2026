from typing import Iterable, Optional

import cv2
import numpy as np

from ..backend.clasificador import Prediction
from ..backend.deteccion import FaceDetection


LANDMARK_DRAW_STEP = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_OK = (40, 210, 145)
COLOR_WARN = (70, 95, 255)


def draw_face_annotations(
    frame: np.ndarray,
    detections: Iterable[FaceDetection],
    predictions: Optional[Iterable[Prediction]] = None,
    show_landmarks: bool = True,
) -> None:
    predictions = list(predictions) if predictions is not None else []
    for idx, detection in enumerate(detections):
        prediction = predictions[idx] if idx < len(predictions) else None
        color = COLOR_OK if prediction and prediction.label != "desconocido" else COLOR_WARN
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, min(y2, y1 + 4)), color, -1)

        if show_landmarks:
            landmark_points = (
                detection.landmarks
                if detection.landmarks_are_sampled
                else detection.landmarks[::LANDMARK_DRAW_STEP]
            )
            for point in landmark_points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 180), -1, cv2.LINE_AA)

        if prediction:
            metric = (
                "similitud coseno"
                if prediction.method.startswith("offline_")
                and prediction.label != "desconocido"
                else ""
            )
            text = f"{prediction.label} {metric} {prediction.confidence * 100:.1f}%".replace("  ", " ")
            draw_label(frame, text, x1, max(32, y1 - 14), color)


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, 0.62, 2)
    y0 = max(4, y - text_h - 12)
    x0 = max(4, x)
    cv2.rectangle(frame, (x0, y0), (x0 + text_w + 18, y0 + text_h + 14), color, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x0 + 9, y0 + text_h + 4), FONT, 0.62, (15, 18, 20), 2, cv2.LINE_AA)

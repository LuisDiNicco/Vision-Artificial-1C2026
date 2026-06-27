from typing import Iterable, Optional

import cv2
import numpy as np

from ..backend.clasificador import Prediction
from ..backend.deteccion import FaceDetection


LANDMARK_DRAW_STEP = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_PANEL = (28, 31, 36)
COLOR_PANEL_2 = (44, 49, 57)
COLOR_TEXT = (240, 244, 248)
COLOR_MUTED = (170, 180, 190)
COLOR_ACCENT = (0, 190, 255)
COLOR_OK = (40, 210, 145)
COLOR_WARN = (70, 95, 255)


def draw_app_chrome(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    status: str,
    controls: Iterable[tuple[str, str]],
) -> None:
    draw_top_bar(frame, title, subtitle)
    draw_control_panel(frame, status, controls)


def draw_face_annotations(
    frame: np.ndarray,
    detections: Iterable[FaceDetection],
    predictions: Optional[Iterable[Prediction]] = None,
) -> None:
    predictions = list(predictions) if predictions is not None else []
    for idx, detection in enumerate(detections):
        prediction = predictions[idx] if idx < len(predictions) else None
        color = COLOR_OK if prediction and prediction.label != "desconocido" else COLOR_WARN
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, min(y2, y1 + 4)), color, -1)

        for point in detection.landmarks[::LANDMARK_DRAW_STEP]:
            cv2.circle(frame, (int(point[0]), int(point[1])), 1, (0, 255, 180), -1, cv2.LINE_AA)

        if prediction:
            text = f"{prediction.label} {prediction.confidence * 100:.1f}%"
            draw_label(frame, text, x1, max(32, y1 - 14), color)


def draw_status_panel(frame: np.ndarray, lines: Iterable[str]) -> None:
    lines = list(lines)
    if not lines:
        return
    h, w = frame.shape[:2]
    panel_h = 28 + 24 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    y = h - panel_h + 28
    for line in lines:
        cv2.putText(frame, line, (14, y), FONT, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
        y += 24


def draw_top_bar(frame: np.ndarray, title: str, subtitle: str) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 64), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, 64), (w, 64), COLOR_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, title, (18, 28), FONT, 0.72, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (18, 52), FONT, 0.45, COLOR_MUTED, 1, cv2.LINE_AA)


def draw_control_panel(frame: np.ndarray, status: str, controls: Iterable[tuple[str, str]]) -> None:
    controls = list(controls)
    h, w = frame.shape[:2]
    panel_h = 92
    y0 = h - panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
    cv2.line(frame, (0, y0), (w, y0), COLOR_ACCENT, 2, cv2.LINE_AA)

    cv2.putText(frame, status[:95], (18, y0 + 28), FONT, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
    x = 18
    y = y0 + 64
    for key, label in controls:
        x = draw_key_hint(frame, key, label, x, y)
        if x > w - 180:
            break


def draw_key_hint(frame: np.ndarray, key: str, label: str, x: int, y: int) -> int:
    key_w = max(34, 12 * len(key) + 16)
    label_w = max(44, 9 * len(label) + 16)
    cv2.rectangle(frame, (x, y - 20), (x + key_w, y + 8), COLOR_ACCENT, -1, cv2.LINE_AA)
    cv2.putText(frame, key, (x + 8, y), FONT, 0.43, (15, 20, 24), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x + key_w, y - 20), (x + key_w + label_w, y + 8), COLOR_PANEL_2, -1)
    cv2.putText(frame, label, (x + key_w + 8, y), FONT, 0.43, COLOR_TEXT, 1, cv2.LINE_AA)
    return x + key_w + label_w + 10


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, 0.62, 2)
    y0 = max(4, y - text_h - 12)
    x0 = max(4, x)
    cv2.rectangle(frame, (x0, y0), (x0 + text_w + 18, y0 + text_h + 14), color, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x0 + 9, y0 + text_h + 4), FONT, 0.62, (15, 18, 20), 2, cv2.LINE_AA)

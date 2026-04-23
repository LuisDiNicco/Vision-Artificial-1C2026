import cv2
import mediapipe as mp
import time
import logging
import os
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Optional, Tuple, TypeVar
import numpy as np


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Modelos oficiales de MediaPipe Tasks (Vision).
HAND_LANDMARKER_MODEL_PATH = 'hand_landmarker.task'
HAND_LANDMARKER_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/latest/hand_landmarker.task'
)

# Topologia de conexiones de mano (21 landmarks) usada por MediaPipe Hands.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def ensure_model(path: str, url: str) -> None:
    """Descarga un modelo Task si no existe en disco."""
    if os.path.exists(path):
        return

    logging.info('Descargando modelo %s', path)
    try:
        urllib.request.urlretrieve(url, path)
        logging.info('Modelo descargado: %s', path)
    except Exception as exc:
        logging.error('No se pudo descargar %s: %s', path, exc)
        logging.error('Continuando con la inicializacion por si el archivo ya existe en otra ruta.')


class CalcState(Enum):
    WAIT_NUM1 = auto()
    WAIT_OPERATOR = auto()
    WAIT_NUM2 = auto()
    SHOW_RESULT = auto()


@dataclass
class CalculatorContext:
    state: CalcState = CalcState.WAIT_NUM1
    num1: Optional[int] = None
    num2: Optional[int] = None
    operator: Optional[str] = None
    result: Optional[float] = None
    error: Optional[str] = None
    last_action_ts: float = 0.0

    def reset(self) -> None:
        self.state = CalcState.WAIT_NUM1
        self.num1 = None
        self.num2 = None
        self.operator = None
        self.result = None
        self.error = None
        self.last_action_ts = time.time()


T = TypeVar('T')


class Stabilizer(Generic[T]):
    """Suaviza detecciones para evitar acciones por ruido entre frames."""

    def __init__(self, size: int = 7, min_ratio: float = 0.72):
        self.history: deque[Optional[T]] = deque(maxlen=size)
        self.min_ratio = min_ratio

    def add(self, value: Optional[T]) -> None:
        self.history.append(value)

    def clear(self) -> None:
        self.history.clear()

    def stable_value(self) -> Optional[T]:
        valid_values = [v for v in self.history if v is not None]
        if not valid_values:
            return None

        counts = Counter(valid_values)
        value, count = counts.most_common(1)[0]
        ratio = count / len(valid_values)
        min_samples = max(4, self.history.maxlen // 2)
        if ratio >= self.min_ratio and len(valid_values) >= min_samples:
            return value
        return None


class HandDigitRecognizer:
    """Reconoce digitos 0-5 a partir de landmarks de mano."""

    # Indices segun la topologia oficial de Hand Landmarker (21 landmarks).
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    INDEX_MCP = 5
    PINKY_MCP = 17
    WRIST = 0
    INDEX_FINGER_TIP = 8
    INDEX_FINGER_PIP = 6
    MIDDLE_FINGER_TIP = 12
    MIDDLE_FINGER_PIP = 10
    RING_FINGER_TIP = 16
    RING_FINGER_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18

    def __init__(self, model_path: str):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.75,
            min_hand_presence_confidence=0.75,
            min_tracking_confidence=0.75,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def detect_digit(self, frame_bgr, timestamp_ms: int) -> Tuple[Optional[int], bool]:
        """Procesa un frame de video y devuelve (digito, mano_presente)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None, False

        hand_landmarks = result.hand_landmarks[0]
        handedness_label = None
        if result.handedness and result.handedness[0]:
            handedness_label = result.handedness[0][0].category_name

        draw_hand_landmarks(frame_bgr, hand_landmarks)

        states = self._finger_states(hand_landmarks, handedness_label)
        digit = self._map_states_to_digit(states)
        return digit, True

    def _finger_states(self, landmarks, handedness_label: Optional[str]) -> Tuple[bool, bool, bool, bool, bool]:
        """Determina si cada dedo esta extendido."""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        index_mcp = landmarks[self.INDEX_MCP]
        pinky_mcp = landmarks[self.PINKY_MCP]
        wrist = landmarks[self.WRIST]

        thumb_margin = 0.018
        if handedness_label == 'Left':
            thumb_open_x = (thumb_tip.x - thumb_ip.x) > thumb_margin
        elif handedness_label == 'Right':
            thumb_open_x = (thumb_ip.x - thumb_tip.x) > thumb_margin
        else:
            thumb_open_x = abs(thumb_tip.x - thumb_ip.x) > thumb_margin

        # Refuerzo para diferenciar mejor 4 vs 5 con mano recta:
        # no alcanza con "pulgar hacia arriba"; tambien debe verse extendido y alejado de la palma.
        thumb_open_y = (thumb_mcp.y - thumb_tip.y) > 0.08

        palm_width = ((index_mcp.x - pinky_mcp.x) ** 2 + (index_mcp.y - pinky_mcp.y) ** 2) ** 0.5
        palm_width = max(palm_width, 1e-6)
        palm_center_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0
        palm_center_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0
        thumb_to_center = ((thumb_tip.x - palm_center_x) ** 2 + (thumb_tip.y - palm_center_y) ** 2) ** 0.5
        thumb_far_from_palm = thumb_to_center > (0.72 * palm_width)

        thumb_tip_to_mcp = ((thumb_tip.x - thumb_mcp.x) ** 2 + (thumb_tip.y - thumb_mcp.y) ** 2) ** 0.5
        thumb_ip_to_mcp = ((thumb_ip.x - thumb_mcp.x) ** 2 + (thumb_ip.y - thumb_mcp.y) ** 2) ** 0.5
        thumb_tip_to_index = ((thumb_tip.x - index_mcp.x) ** 2 + (thumb_tip.y - index_mcp.y) ** 2) ** 0.5

        # Si el pulgar esta doblado sobre la palma, esta razon baja.
        thumb_reach = thumb_tip_to_mcp > (1.22 * max(thumb_ip_to_mcp, 1e-6))
        thumb_away_from_index = thumb_tip_to_index > (0.50 * palm_width)

        # Medimos rectitud del pulgar (MCP->IP->TIP) con coseno de angulo.
        v1x, v1y = (thumb_ip.x - thumb_mcp.x), (thumb_ip.y - thumb_mcp.y)
        v2x, v2y = (thumb_tip.x - thumb_ip.x), (thumb_tip.y - thumb_ip.y)
        n1 = (v1x * v1x + v1y * v1y) ** 0.5
        n2 = (v2x * v2x + v2y * v2y) ** 0.5
        cos_angle = (v1x * v2x + v1y * v2y) / max(n1 * n2, 1e-6)
        thumb_straight = cos_angle > 0.45

        thumb_open = (
            (thumb_open_x or thumb_open_y)
            and thumb_far_from_palm
            and thumb_reach
            and thumb_away_from_index
            and thumb_straight
        )

        index_open = landmarks[self.INDEX_FINGER_TIP].y < landmarks[self.INDEX_FINGER_PIP].y
        middle_open = landmarks[self.MIDDLE_FINGER_TIP].y < landmarks[self.MIDDLE_FINGER_PIP].y
        ring_open = landmarks[self.RING_FINGER_TIP].y < landmarks[self.RING_FINGER_PIP].y
        pinky_open = landmarks[self.PINKY_TIP].y < landmarks[self.PINKY_PIP].y

        return thumb_open, index_open, middle_open, ring_open, pinky_open

    def _map_states_to_digit(self, states: Tuple[bool, bool, bool, bool, bool]) -> Optional[int]:
        """Mapea patron de dedos a digitos 0-5."""
        thumb, index_, middle, ring, pinky = states

        if not index_ and not middle and not ring and not pinky:
            return 0
        if index_ and not middle and not ring and not pinky:
            return 1
        if index_ and middle and not ring and not pinky:
            return 2
        if index_ and middle and ring and not pinky:
            return 3
        if index_ and middle and ring and pinky and not thumb:
            return 4
        if thumb and index_ and middle and ring and pinky:
            return 5
        return None


def draw_hand_landmarks(frame_bgr, hand_landmarks) -> None:
    """Dibuja conexiones y puntos de la mano sin depender de módulos internos."""
    h, w = frame_bgr.shape[:2]

    # Conexiones oficiales de la topologia de mano.
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        x0, y0 = int(start.x * w), int(start.y * h)
        x1, y1 = int(end.x * w), int(end.y * h)
        cv2.line(frame_bgr, (x0, y0), (x1, y1), (0, 220, 0), 2)

    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame_bgr, (x, y), 4, (0, 255, 255), -1)


def compute_result(a: int, op: str, b: int) -> Tuple[Optional[float], Optional[str]]:
    if op == '+':
        return float(a + b), None
    if op == '-':
        return float(a - b), None
    if op == '*':
        return float(a * b), None
    if op == '/':
        if b == 0:
            return None, 'Error: division por cero'
        return a / b, None
    return None, 'Error: operador invalido'


def format_result(value: Optional[float]) -> str:
    if value is None:
        return '-'
    if abs(value - int(value)) < 1e-9:
        return str(int(value))
    return f'{value:.2f}'


def lighting_warning(frame_bgr) -> Optional[str]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean()
    if mean_intensity < 45:
        return 'Advertencia: poca luz'
    if mean_intensity > 215:
        return 'Advertencia: demasiada luz'
    return None


def wrap_text_to_width(
    text: str,
    max_width: int,
    font_face: int,
    font_scale: float,
    thickness: int,
) -> list[str]:
    """Corta una linea en varias para que entre en un ancho maximo."""
    words = text.split()
    if not words:
        return ['']

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f'{current} {word}'
        candidate_width = cv2.getTextSize(candidate, font_face, font_scale, thickness)[0][0]
        if candidate_width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_overlay(
    frame,
    fps: float,
    context: CalculatorContext,
    current_digit: Optional[int],
    stable_digit: Optional[int],
    status_msg: str,
    light_msg: Optional[str],
    hand_missing_seconds: float,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Construir una vista compuesta: camara completa + panel separado a la derecha.
    panel_w = min(520, max(360, int(w * 0.46)))
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame

    panel_x0 = w
    panel_x1 = w + panel_w
    panel_pad = 20
    content_x = panel_x0 + panel_pad
    content_w = panel_w - (panel_pad * 2)

    cv2.rectangle(canvas, (panel_x0, 0), (panel_x1 - 1, h - 1), (20, 20, 20), -1)
    cv2.rectangle(canvas, (panel_x0 + 8, 8), (panel_x1 - 9, h - 9), (80, 180, 80), 1)

    lines = [
        f'FPS: {fps:.1f}',
        f'Estado: {context.state.name}',
        f'Digito actual: {current_digit if current_digit is not None else "-"}',
        f'Digito estable: {stable_digit if stable_digit is not None else "-"}',
        f'Operacion: {context.num1 if context.num1 is not None else "_"} '
        f'{context.operator if context.operator is not None else "_"} '
        f'{context.num2 if context.num2 is not None else "_"}',
        f'Resultado: {format_result(context.result)}',
    ]

    y = 36
    for text in lines:
        cv2.putText(canvas, text, (content_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 255, 230), 1)
        y += 30

    if status_msg:
        cv2.putText(
            canvas,
            'Estado actual:',
            (content_x, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 220, 255),
            1,
        )
        status_lines = wrap_text_to_width(
            status_msg,
            max_width=content_w,
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.66,
            thickness=2,
        )
        status_y = y + 40
        for line in status_lines:
            cv2.putText(
                canvas,
                line,
                (content_x, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (80, 230, 255),
                2,
            )
            status_y += 30

    if context.error:
        error_lines = wrap_text_to_width(
            context.error,
            max_width=content_w,
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.58,
            thickness=2,
        )
        error_y = y + 74
        for line in error_lines:
            cv2.putText(canvas, line, (content_x, error_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2)
            error_y += 24

    if light_msg:
        light_lines = wrap_text_to_width(
            light_msg,
            max_width=content_w,
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.54,
            thickness=2,
        )
        light_y = h - 94
        for line in light_lines:
            cv2.putText(canvas, line, (content_x, light_y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 180, 255), 2)
            light_y += 22

    if hand_missing_seconds > 1.2:
        cv2.putText(
            canvas,
            f'Mano no detectada ({hand_missing_seconds:.1f}s)',
            (content_x, h - 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (0, 190, 255),
            2,
        )

    keys_text = 'Teclas: Enter=confirmar | + - * / = operador | r=reset | q=salir'
    key_lines = wrap_text_to_width(
        keys_text,
        max_width=content_w,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.50,
        thickness=1,
    )
    key_y = h - 22 - (len(key_lines) - 1) * 20
    for line in key_lines:
        cv2.putText(
            canvas,
            line,
            (content_x, key_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )
        key_y += 20

    return canvas


def try_capture_number(context: CalculatorContext, stable_digit: Optional[int]) -> str:
    if stable_digit is None:
        return 'No hay digito estable para confirmar'

    context.error = None
    if context.state == CalcState.WAIT_NUM1:
        context.num1 = stable_digit
        context.state = CalcState.WAIT_OPERATOR
        context.last_action_ts = time.time()
        logging.info('NUM1 confirmado: %s', stable_digit)
        return f'NUM1 confirmado: {stable_digit}'

    if context.state == CalcState.WAIT_NUM2:
        context.num2 = stable_digit
        result, err = compute_result(context.num1, context.operator, context.num2)
        context.result = result
        context.error = err
        context.state = CalcState.SHOW_RESULT
        context.last_action_ts = time.time()
        if err:
            logging.warning('Fallo al calcular: %s', err)
            return err
        logging.info(
            'Resultado calculado: %s %s %s = %s',
            context.num1,
            context.operator,
            context.num2,
            format_result(result),
        )
        return f'Resultado: {context.num1} {context.operator} {context.num2} = {format_result(result)}'

    if context.state == CalcState.SHOW_RESULT:
        return 'Resultado mostrado. Presiona r para reiniciar'

    return 'Primero selecciona operador (+ - * /)'


def try_set_operator(context: CalculatorContext, op: str) -> str:
    if context.state != CalcState.WAIT_OPERATOR:
        return 'Operador fuera de secuencia'

    context.operator = op
    context.state = CalcState.WAIT_NUM2
    context.error = None
    context.last_action_ts = time.time()
    logging.info('Operador seleccionado: %s', op)
    return f'Operador seleccionado: {op}'


def main() -> None:
    ensure_model(HAND_LANDMARKER_MODEL_PATH, HAND_LANDMARKER_MODEL_URL)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('Error: no se pudo abrir la webcam')
        return

    hand_digit_recognizer = HandDigitRecognizer(HAND_LANDMARKER_MODEL_PATH)

    digit_stabilizer = Stabilizer[int](size=7, min_ratio=0.72)

    context = CalculatorContext()
    context.reset()

    status_msg = 'Mostra una mano (0-5) y confirma con Enter.'

    last_frame_ts = time.time()
    last_hand_seen_ts = time.time()
    last_confirm_ts = 0.0

    hand_lost_reset_sec = 6.0
    inactivity_reset_sec = 35.0
    confirm_cooldown_sec = 0.6

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        timestamp_ms = int(now * 1000)
        dt = now - last_frame_ts
        fps = 1.0 / dt if dt > 0 else 0.0
        last_frame_ts = now

        current_digit, hand_present = hand_digit_recognizer.detect_digit(frame, timestamp_ms)
        if hand_present:
            last_hand_seen_ts = now

        digit_stabilizer.add(current_digit if hand_present else None)
        stable_digit = digit_stabilizer.stable_value()

        missing_sec = now - last_hand_seen_ts
        if missing_sec > hand_lost_reset_sec and context.state != CalcState.WAIT_NUM1:
            context.reset()
            digit_stabilizer.clear()
            status_msg = 'Reset automatico por perdida de mano'
            logging.info(status_msg)
            last_hand_seen_ts = now

        if (now - context.last_action_ts) > inactivity_reset_sec and context.state != CalcState.WAIT_NUM1:
            context.reset()
            digit_stabilizer.clear()
            status_msg = 'Reset automatico por inactividad'
            logging.info(status_msg)

        light_msg = lighting_warning(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('r'):
            context.reset()
            digit_stabilizer.clear()
            status_msg = 'Reset manual realizado'

        if key in (13, 10):
            if (now - last_confirm_ts) >= confirm_cooldown_sec:
                status_msg = try_capture_number(context, stable_digit)
                last_confirm_ts = now
            else:
                status_msg = 'Espera un instante antes de confirmar de nuevo'

        if key in (ord('+'), ord('-'), ord('*'), ord('/')):
            status_msg = try_set_operator(context, chr(key))

        ui_frame = draw_overlay(
            frame=frame,
            fps=fps,
            context=context,
            current_digit=current_digit,
            stable_digit=stable_digit,
            status_msg=status_msg,
            light_msg=light_msg,
            hand_missing_seconds=missing_sec,
        )

        cv2.imshow('TP 1 v2 - Calculadora por Mano (0-5)', ui_frame)

    hand_digit_recognizer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

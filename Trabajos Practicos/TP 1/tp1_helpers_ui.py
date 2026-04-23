import cv2
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Optional, TypeVar

import numpy as np


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


def compute_result(a: int, op: str, b: int):
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

    # Vista compuesta: camara completa + panel separado a la derecha.
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

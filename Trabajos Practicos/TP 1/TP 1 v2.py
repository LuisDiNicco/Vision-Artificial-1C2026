import cv2
import mediapipe as mp
import time
import logging
import os
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Descargar modelo si no existe
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'

if not os.path.exists(MODEL_PATH):
    print(f'Descargando modelo {MODEL_PATH}...')
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f'Modelo {MODEL_PATH} descargado correctamente.')
    except Exception as e:
        print(f'Error al descargar modelo: {e}')
        print('El programa continuará intentando usar el modelo.')


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


class Stabilizer:
    def __init__(self, size: int = 7, min_ratio: float = 0.72):
        self.history = deque(maxlen=size)
        self.min_ratio = min_ratio

    def add(self, value: Optional[int]) -> None:
        self.history.append(value)

    def clear(self) -> None:
        self.history.clear()

    def stable_value(self) -> Optional[int]:
        valid_values = [v for v in self.history if v is not None]
        if not valid_values:
            return None

        counts = Counter(valid_values)
        value, count = counts.most_common(1)[0]
        # Usar solo frames validos evita castigar la estabilidad cuando hay perdidas breves de mano.
        ratio = count / len(valid_values)
        if ratio >= self.min_ratio and len(valid_values) >= max(4, self.history.maxlen // 2):
            return value
        return None


class HandDigitRecognizer:
    # Índices de landmarks según MediaPipe Hand Landmarks (21 puntos)
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    INDEX_FINGER_TIP = 8
    INDEX_FINGER_PIP = 6
    MIDDLE_FINGER_TIP = 12
    MIDDLE_FINGER_PIP = 10
    RING_FINGER_TIP = 16
    RING_FINGER_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18
    WRIST = 0

    def __init__(self):
        """Inicializa el reconocedor de mano usando MediaPipe Tasks API (0.10.33+)"""
        try:
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.75,
                min_hand_presence_confidence=0.75,
                min_tracking_confidence=0.75,
            )
            self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
            logging.info('HandLandmarker inicializado correctamente')
        except Exception as e:
            logging.error(f'Error al inicializar HandLandmarker: {e}')
            raise

    def close(self) -> None:
        """Cierra el landmarker"""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()

    def _finger_states(self, landmarks, handedness_label: Optional[str]) -> Tuple[bool, bool, bool, bool, bool]:
        """Detecta el estado de cada dedo (extendido o plegado)"""
        # Para el pulgar usamos una regla mixta (X con lateralidad + fallback en Y)
        # porque en camaras frontales el eje X puede ser ruidoso segun angulo de mano.
        thumb_tip_x = landmarks[self.THUMB_TIP].x
        thumb_ip_x = landmarks[self.THUMB_IP].x
        thumb_tip_y = landmarks[self.THUMB_TIP].y
        thumb_mcp_y = landmarks[self.THUMB_MCP].y
        thumb_margin = 0.01
        if handedness_label == 'Left':
            thumb_open_x = (thumb_tip_x - thumb_ip_x) > thumb_margin
        else:
            thumb_open_x = (thumb_ip_x - thumb_tip_x) > thumb_margin

        # Fallback: si el pulgar esta claramente levantado, aceptarlo como abierto.
        thumb_open_y = (thumb_mcp_y - thumb_tip_y) > 0.04
        thumb_open = thumb_open_x or thumb_open_y

        # Otros dedos: punta por encima de la articulación PIP
        index_open = landmarks[self.INDEX_FINGER_TIP].y < landmarks[self.INDEX_FINGER_PIP].y
        middle_open = landmarks[self.MIDDLE_FINGER_TIP].y < landmarks[self.MIDDLE_FINGER_PIP].y
        ring_open = landmarks[self.RING_FINGER_TIP].y < landmarks[self.RING_FINGER_PIP].y
        pinky_open = landmarks[self.PINKY_TIP].y < landmarks[self.PINKY_PIP].y

        return thumb_open, index_open, middle_open, ring_open, pinky_open

    def _map_states_to_digit(self, states: Tuple[bool, bool, bool, bool, bool]) -> Optional[int]:
        """Mapea los estados de los dedos a un dígito 0-5"""
        thumb, index_, middle, ring, pinky = states

        # Para 0-4 ignoramos pulgar (es el dedo mas ruidoso en camara frontal).
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

        # Para 5 exigimos los cuatro dedos largos abiertos y pulgar abierto.
        if thumb and index_ and middle and ring and pinky:
            return 5
        return None

    def detect_digit(self, frame_bgr) -> Tuple[Optional[int], bool]:
        """Detecta el dígito en el frame actual"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            result = self.landmarker.detect_for_video(mp_image, int(time.time() * 1000))
        except Exception as e:
            logging.warning(f'Error al detectar: {e}')
            return None, False

        if not result.hand_landmarks:
            return None, False

        hand_landmarks = result.hand_landmarks[0]
        handedness_label = None
        if getattr(result, 'handedness', None) and result.handedness[0]:
            handedness_label = result.handedness[0][0].category_name

        # Dibujar landmarks para feedback visual
        self._draw_landmarks(frame_bgr, hand_landmarks)

        states = self._finger_states(hand_landmarks, handedness_label)
        digit = self._map_states_to_digit(states)
        return digit, True

    def _draw_landmarks(self, frame, landmarks):
        """Dibuja los landmarks de la mano en el frame"""
        h_frame, w_frame = frame.shape[:2]
        
        # Dibujar puntos principales de los dedos
        key_points = [
            self.INDEX_FINGER_TIP, self.MIDDLE_FINGER_TIP, self.RING_FINGER_TIP,
            self.PINKY_TIP, self.THUMB_TIP, self.WRIST
        ]
        
        for point_idx in key_points:
            lm = landmarks[point_idx]
            x, y = int(lm.x * w_frame), int(lm.y * h_frame)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)



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


def draw_overlay(
    frame,
    fps: float,
    context: CalculatorContext,
    current_digit: Optional[int],
    stable_digit: Optional[int],
    status_msg: str,
    light_msg: Optional[str],
    hand_missing_seconds: float,
) -> None:
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (10, 10), (w - 10, 210), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (w - 10, 210), (80, 180, 80), 1)

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

    y = 35
    for text in lines:
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 255, 230), 2)
        y += 28

    if context.error:
        cv2.putText(frame, context.error, (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)

    cv2.putText(
        frame,
        'Teclas: Enter=confirmar numero | + - * / = operador | r=reset | q=salir',
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    if status_msg:
        cv2.putText(frame, status_msg, (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 230, 255), 2)

    if light_msg:
        cv2.putText(frame, light_msg, (20, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 180, 255), 2)

    if hand_missing_seconds > 1.2:
        cv2.putText(
            frame,
            f'Mano no detectada ({hand_missing_seconds:.1f}s)',
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 190, 255),
            2,
        )


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
        logging.info('Resultado calculado: %s %s %s = %s', context.num1, context.operator, context.num2, format_result(result))
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
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('Error: no se pudo abrir la webcam')
        return

    recognizer = HandDigitRecognizer()
    stabilizer = Stabilizer(size=7, min_ratio=0.72)
    context = CalculatorContext()
    context.reset()

    status_msg = 'Mostra una mano con numeros 0-5 y confirma con Enter'

    last_frame_ts = time.time()
    last_hand_seen_ts = time.time()
    last_confirm_ts = 0.0

    HAND_LOST_RESET_SEC = 6.0
    INACTIVITY_RESET_SEC = 35.0
    CONFIRM_COOLDOWN_SEC = 0.6

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        dt = now - last_frame_ts
        fps = 1.0 / dt if dt > 0 else 0.0
        last_frame_ts = now

        current_digit, hand_present = recognizer.detect_digit(frame)
        if hand_present:
            last_hand_seen_ts = now

        stabilizer.add(current_digit if hand_present else None)
        stable_digit = stabilizer.stable_value()

        missing_sec = now - last_hand_seen_ts
        if missing_sec > HAND_LOST_RESET_SEC and context.state != CalcState.WAIT_NUM1:
            context.reset()
            stabilizer.clear()
            status_msg = 'Reset automatico por perdida de mano'
            logging.info(status_msg)
            last_hand_seen_ts = now

        if (now - context.last_action_ts) > INACTIVITY_RESET_SEC and context.state != CalcState.WAIT_NUM1:
            context.reset()
            stabilizer.clear()
            status_msg = 'Reset automatico por inactividad'
            logging.info(status_msg)

        light_msg = lighting_warning(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            context.reset()
            stabilizer.clear()
            status_msg = 'Reset manual realizado'

        if key in (13, 10):
            if (now - last_confirm_ts) >= CONFIRM_COOLDOWN_SEC:
                status_msg = try_capture_number(context, stable_digit)
                last_confirm_ts = now
            else:
                status_msg = 'Espera un instante antes de confirmar de nuevo'

        if key in (ord('+'), ord('-'), ord('*'), ord('/')):
            status_msg = try_set_operator(context, chr(key))

        draw_overlay(
            frame=frame,
            fps=fps,
            context=context,
            current_digit=current_digit,
            stable_digit=stable_digit,
            status_msg=status_msg,
            light_msg=light_msg,
            hand_missing_seconds=missing_sec,
        )

        cv2.imshow('TP 1 v2 - Calculadora por Mano (0-5)', frame)

    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

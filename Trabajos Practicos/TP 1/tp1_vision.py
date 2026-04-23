import cv2
import mediapipe as mp
import logging
import os
import urllib.request
from typing import Optional, Tuple

# Modelo oficial de MediaPipe Tasks (Vision) para mano.
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
        # Clases base de MediaPipe Tasks para crear el detector de mano.
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Configura el detector en modo VIDEO (requiere timestamp por frame).
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.75,
            min_hand_presence_confidence=0.75,
            min_tracking_confidence=0.75,
        )
        # Instancia final del detector que se usa en cada frame.
        self._landmarker = HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def detect_digit(self, frame_bgr, timestamp_ms: int) -> Tuple[Optional[int], bool]:
        """Procesa un frame de video y devuelve (digito, mano_presente)."""
        # OpenCV entrega BGR, pero MediaPipe espera RGB.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Empaqueta la imagen en el contenedor de MediaPipe Tasks.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Ejecuta inferencia de landmarks para este instante de video.
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None, False

        # Toma la primera mano detectada (el sistema esta configurado en num_hands=1).
        hand_landmarks = result.hand_landmarks[0]
        handedness_label = None
        if result.handedness and result.handedness[0]:
            # Left/Right segun clasificador de lateralidad de MediaPipe.
            handedness_label = result.handedness[0][0].category_name

        # Dibuja la topologia de la mano sobre el frame original.
        draw_hand_landmarks(frame_bgr, hand_landmarks)

        # Convierte landmarks en estados booleanos de dedos y luego en un digito.
        states = self._finger_states(hand_landmarks, handedness_label)
        digit = self._map_states_to_digit(states)
        return digit, True

    def _finger_states(self, landmarks, handedness_label: Optional[str]) -> Tuple[bool, bool, bool, bool, bool]:
        """Determina si cada dedo esta extendido."""
        # Landmarks clave para medir apertura del pulgar y geometria de palma.
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        index_mcp = landmarks[self.INDEX_MCP]
        pinky_mcp = landmarks[self.PINKY_MCP]
        wrist = landmarks[self.WRIST]

        # Apertura lateral del pulgar segun mano izquierda/derecha.
        thumb_margin = 0.018
        if handedness_label == 'Left':
            thumb_open_x = (thumb_tip.x - thumb_ip.x) > thumb_margin
        elif handedness_label == 'Right':
            thumb_open_x = (thumb_ip.x - thumb_tip.x) > thumb_margin
        else:
            thumb_open_x = abs(thumb_tip.x - thumb_ip.x) > thumb_margin

        # Refuerzo para diferenciar mejor 4 vs 5 con mano recta.
        # Si el pulgar apunta hacia arriba, hay mayor chance de estar abierto.
        thumb_open_y = (thumb_mcp.y - thumb_tip.y) > 0.08

        # Estima ancho de palma y centro para normalizar distancias.
        palm_width = ((index_mcp.x - pinky_mcp.x) ** 2 + (index_mcp.y - pinky_mcp.y) ** 2) ** 0.5
        palm_width = max(palm_width, 1e-6)
        palm_center_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0
        palm_center_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0
        thumb_to_center = ((thumb_tip.x - palm_center_x) ** 2 + (thumb_tip.y - palm_center_y) ** 2) ** 0.5
        # El pulgar debe estar suficientemente lejos de la palma para contar como abierto.
        thumb_far_from_palm = thumb_to_center > (0.72 * palm_width)

        # Distancias auxiliares para evitar confundir 4 con 5.
        thumb_tip_to_mcp = ((thumb_tip.x - thumb_mcp.x) ** 2 + (thumb_tip.y - thumb_mcp.y) ** 2) ** 0.5
        thumb_ip_to_mcp = ((thumb_ip.x - thumb_mcp.x) ** 2 + (thumb_ip.y - thumb_mcp.y) ** 2) ** 0.5
        thumb_tip_to_index = ((thumb_tip.x - index_mcp.x) ** 2 + (thumb_tip.y - index_mcp.y) ** 2) ** 0.5

        # Si el pulgar esta plegado, esta razon baja notablemente.
        thumb_reach = thumb_tip_to_mcp > (1.22 * max(thumb_ip_to_mcp, 1e-6))
        # Evita aceptar pulgar cuando queda pegado al indice.
        thumb_away_from_index = thumb_tip_to_index > (0.50 * palm_width)

        # Calcula rectitud del pulgar con el angulo MCP->IP->TIP.
        v1x, v1y = (thumb_ip.x - thumb_mcp.x), (thumb_ip.y - thumb_mcp.y)
        v2x, v2y = (thumb_tip.x - thumb_ip.x), (thumb_tip.y - thumb_ip.y)
        n1 = (v1x * v1x + v1y * v1y) ** 0.5
        n2 = (v2x * v2x + v2y * v2y) ** 0.5
        cos_angle = (v1x * v2x + v1y * v2y) / max(n1 * n2, 1e-6)
        thumb_straight = cos_angle > 0.45

        # Pulgar abierto final: combinacion de reglas para robustez.
        thumb_open = (
            (thumb_open_x or thumb_open_y)
            and thumb_far_from_palm
            and thumb_reach
            and thumb_away_from_index
            and thumb_straight
        )

        # Para los otros dedos: punta arriba de PIP => dedo extendido.
        index_open = landmarks[self.INDEX_FINGER_TIP].y < landmarks[self.INDEX_FINGER_PIP].y
        middle_open = landmarks[self.MIDDLE_FINGER_TIP].y < landmarks[self.MIDDLE_FINGER_PIP].y
        ring_open = landmarks[self.RING_FINGER_TIP].y < landmarks[self.RING_FINGER_PIP].y
        pinky_open = landmarks[self.PINKY_TIP].y < landmarks[self.PINKY_PIP].y

        return thumb_open, index_open, middle_open, ring_open, pinky_open

    def _map_states_to_digit(self, states: Tuple[bool, bool, bool, bool, bool]) -> Optional[int]:
        """Mapea patron de dedos a digitos 0-5."""
        thumb, index_, middle, ring, pinky = states

        # Mapeo explicito de combinaciones para 0..5.
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
    # Medidas del frame para pasar de coordenadas normalizadas [0..1] a pixeles.
    h, w = frame_bgr.shape[:2]

    # Dibuja lineas de la topologia oficial de mano (esqueleto).
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        x0, y0 = int(start.x * w), int(start.y * h)
        x1, y1 = int(end.x * w), int(end.y * h)
        cv2.line(frame_bgr, (x0, y0), (x1, y1), (0, 220, 0), 2)

    # Dibuja cada landmark como punto para referencia visual.
    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame_bgr, (x, y), 4, (0, 255, 255), -1)

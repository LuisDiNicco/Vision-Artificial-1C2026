import cv2
import logging
import time

from tp1_helpers_ui import (
    CalcState,
    CalculatorContext,
    Stabilizer,
    draw_overlay,
    lighting_warning,
    try_capture_number,
    try_set_operator,
)
from tp1_vision import (
    HAND_LANDMARKER_MODEL_PATH,
    HAND_LANDMARKER_MODEL_URL,
    HandDigitRecognizer,
    ensure_model,
)


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def main() -> None:
    # Verifica si el modelo de mano existe localmente y lo descarga si hace falta.
    ensure_model(HAND_LANDMARKER_MODEL_PATH, HAND_LANDMARKER_MODEL_URL)

    # Inicializa la webcam principal del sistema (indice 0).
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('Error: no se pudo abrir la webcam')
        return

    # Crea el detector de digitos basado en Hand Landmarker.
    hand_digit_recognizer = HandDigitRecognizer(HAND_LANDMARKER_MODEL_PATH)
    # Suaviza detecciones entre frames para evitar saltos por ruido.
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
        # Lee un frame BGR de la webcam en cada iteracion.
        ok, frame = cap.read()
        if not ok:
            break

        # Marca temporal para API de video de MediaPipe y calculo de FPS.
        now = time.time()
        timestamp_ms = int(now * 1000)
        dt = now - last_frame_ts
        fps = 1.0 / dt if dt > 0 else 0.0
        last_frame_ts = now

        # Paso principal de vision: detecta landmarks de mano y mapea un digito 0-5.
        current_digit, hand_present = hand_digit_recognizer.detect_digit(frame, timestamp_ms)
        if hand_present:
            last_hand_seen_ts = now

        # Acumula historial de digitos para obtener una lectura estable.
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

        # Genera la salida visual final: frame de camara + panel lateral de estado.
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

        # Muestra la interfaz en tiempo real.
        cv2.imshow('TP 1 Grupo 4 - Calculadora por Mano (0-5)', ui_frame)

    # Libera recursos nativos de vision y ventanas OpenCV.
    hand_digit_recognizer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

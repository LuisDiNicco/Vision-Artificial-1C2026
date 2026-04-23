import mediapipe as mp
import cv2
import urllib.request
import os
import time

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# =========================
# Configuracion de modelos
# =========================
# Estos modelos son oficiales de MediaPipe para deteccion de objetos (COCO).
MODEL_OPTIONS = {
    '0': {
        'name': 'EfficientDet-Lite0 (float16)',
        'path': 'efficientdet_lite0.tflite',
        'url': 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite',
    },
    '2': {
        'name': 'EfficientDet-Lite2 (float16)',
        'path': 'efficientdet_lite2.tflite',
        'url': 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/latest/efficientdet_lite2.tflite',
    },
}
DEFAULT_MODEL_KEY = '0'

def ensure_model(model_key: str):
    """Descarga el modelo seleccionado si no existe localmente."""
    model_data = MODEL_OPTIONS[model_key]
    model_path = model_data['path']
    model_url = model_data['url']

    if not os.path.exists(model_path):
        print(f"Descargando {model_data['name']}...")
        urllib.request.urlretrieve(model_url, model_path)
        print('Modelo descargado.')

    return model_path, model_data['name']


def create_detector(model_path: str):
    """Crea el detector en modo LIVE_STREAM (igual que en los ejemplos de Google)."""
    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        max_results=5,
        score_threshold=0.5,
        result_callback=save_result)

    return ObjectDetector.create_from_options(options)

# Variable global para almacenar el ultimo resultado de deteccion
latest_result = None

def save_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

selected_model_key = DEFAULT_MODEL_KEY
model_path, model_name = ensure_model(selected_model_key)

with create_detector(model_path) as detector:
    # Abrir webcam (CAP_DSHOW suele funcionar mejor en Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print('Error: no se pudo abrir la webcam')
        exit(1)

    frame_index = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # Convertir BGR (OpenCV) a RGB (MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # frame_index como timestamp creciente para detect_async
        detector.detect_async(mp_image, frame_index)

        detections_count = 0

        # Dibujar detecciones disponibles del callback asincronico
        if latest_result is not None:
            detections_count = len(latest_result.detections)
            for detection in latest_result.detections:
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

                # Rectangulo verde
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Nombre de clase + porcentaje de confianza
                category = detection.categories[0]
                label = f'{category.category_name} ({category.score:.0%})'
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS simple para monitorear rendimiento
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # Panel de informacion para exponer estado del detector
        cv2.rectangle(frame, (10, 10), (760, 75), (0, 0, 0), -1)
        cv2.putText(frame, f'Modelo: {model_name}', (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f'FPS: {fps:.1f} | Detecciones: {detections_count}', (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Detector de Objetos (TP 1 Grupo 4)', frame)

        # q: salir | 0: Lite0 | 2: Lite2 (reinicia detector)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in (ord('0'), ord('2')):
            requested_key = chr(key)
            if requested_key != selected_model_key:
                selected_model_key = requested_key
                model_path, model_name = ensure_model(selected_model_key)
                detector.close()
                detector = create_detector(model_path)
                latest_result = None

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

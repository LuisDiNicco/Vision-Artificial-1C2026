import mediapipe as mp
import cv2
import urllib.request
import os

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Descargar modelo si no existe localmente
model_path = 'efficientdet_lite0.tflite'
model_url = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite'

if not os.path.exists(model_path):
    print('Descargando modelo...')
    urllib.request.urlretrieve(model_url, model_path)
    print('Modelo descargado.')

# Variable global para almacenar el ultimo resultado de deteccion
latest_result = None

def save_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    score_threshold=0.5,
    result_callback=save_result)

with ObjectDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print('Error: no se pudo abrir la webcam')
        exit(1)

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Usar frame_index como timestamp para garantizar valores crecientes
        detector.detect_async(mp_image, frame_index)

        # Dibujar detecciones en el frame
        if latest_result is not None:
            for detection in latest_result.detections:
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

                # Rectangulo verde
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Nombre y confianza
                category = detection.categories[0]
                label = f'{category.category_name} ({category.score:.0%})'
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Detector de Objetos', frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

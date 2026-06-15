import argparse
from collections import deque

import cv2

from ..backend.alineamiento import align_face
from ..backend.camera import list_available_cameras, next_available_camera, open_webcam
from ..backend.clasificador import load_classifier
from ..backend.data import MODEL_PATH
from ..backend.deteccion import MediaPipeFaceDetector
from ..backend.embeddings import ArcFaceEmbedder
from ..frontend.opencv_ui import draw_app_chrome, draw_face_annotations


SMOOTHING_WINDOW = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconocimiento facial en tiempo real.")
    parser.add_argument("--camera", type=int, default=0, help="Indice de camara OpenCV a usar.")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Lista indices de camara disponibles y termina.",
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=8,
        help="Indice maximo a probar con --list-cameras o al cambiar con C.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        cameras = list_available_cameras(args.max_camera_index)
        print(f"Camaras disponibles: {cameras if cameras else 'ninguna'}")
        return

    if not MODEL_PATH.exists():
        print("No existe el modelo entrenado.")
        print("Primero ejecuta: python tp_integrador_registro.py")
        return

    classifier = load_classifier(MODEL_PATH)
    detector = MediaPipeFaceDetector(max_faces=4)
    embedder = ArcFaceEmbedder()

    cap = open_webcam(args.camera)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la webcam con indice {args.camera}.")
        detector.close()
        return

    status = "Reconocimiento activo"
    camera_index = args.camera
    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        detections = detector.detect(frame)

        predictions = []
        for detection in detections:
            aligned = align_face(frame, detection)
            embedding = embedder.embed(aligned)
            predictions.append(classifier.predict(embedding))

        if len(predictions) == 1:
            recent_predictions.append(predictions[0])
            predictions[0] = smooth_single_prediction(recent_predictions)
        else:
            recent_predictions.clear()

        draw_face_annotations(frame, detections, predictions)
        draw_app_chrome(
            frame,
            "TP Integrador - Reconocimiento",
            f"Camara: {camera_index} | Rostros: {len(detections)} | Umbral desconocido: {classifier.distance_threshold:.2f}",
            status,
            [
                ("C", "camara"),
                ("Q", "salir"),
            ],
        )
        cv2.imshow("TP Integrador - Reconocimiento Facial", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("c"), ord("C")):
            next_index, next_cap = next_available_camera(camera_index, args.max_camera_index)
            if next_cap is None:
                status = "No se encontro otra camara disponible."
            else:
                cap.release()
                cap = next_cap
                camera_index = next_index
                status = f"Camara cambiada a indice {camera_index}."

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def smooth_single_prediction(recent_predictions):
    if not recent_predictions:
        return None

    grouped = {}
    for prediction in recent_predictions:
        grouped.setdefault(prediction.label, []).append(prediction)

    label = max(
        grouped,
        key=lambda key: (len(grouped[key]), sum(item.confidence for item in grouped[key])),
    )
    group = grouped[label]
    confidence = sum(item.confidence for item in group) / len(group)
    distance = sum(item.distance for item in group) / len(group)
    method = group[-1].method
    return type(group[-1])(label=label, confidence=confidence, distance=distance, method=method)


if __name__ == "__main__":
    main()

import argparse
import time

import cv2

from ..backend.alineamiento import align_face
from ..backend.camera import list_available_cameras, next_available_camera, open_webcam
from ..backend.clasificador import save_classifier, train_classifier
from ..backend.data import MODEL_PATH, load_embeddings, save_sample
from ..backend.deteccion import MediaPipeFaceDetector, largest_face
from ..backend.embeddings import ArcFaceEmbedder
from ..frontend.opencv_ui import draw_app_chrome, draw_face_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Registro de rostros para TP Integrador.")
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


def train_and_save() -> str:
    embeddings, labels = load_embeddings()
    if len(labels) < 2:
        return "Necesitas al menos 2 capturas para entrenar."
    classifier = train_classifier(embeddings, labels)
    save_classifier(classifier, MODEL_PATH)
    people = ", ".join(sorted(set(labels)))
    return f"Modelo entrenado con {len(labels)} muestras: {people}"


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        cameras = list_available_cameras(args.max_camera_index)
        print(f"Camaras disponibles: {cameras if cameras else 'ninguna'}")
        return

    print("Registro de rostros con embeddings ArcFace")
    print("Las capturas se guardan en datos_privados/ y no se suben al repo.")
    save_photos = input("Guardar tambien fotos alineadas? [s/N]: ").strip().lower() == "s"
    current_name = input("Nombre y apellido de la persona: ").strip()
    while not current_name:
        current_name = input("Ingresa un nombre valido: ").strip()

    cap = open_webcam(args.camera)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la webcam con indice {args.camera}.")
        return

    detector = MediaPipeFaceDetector(max_faces=1)
    embedder = ArcFaceEmbedder()
    status = f"Registrando: {current_name}"
    camera_index = args.camera
    count_current = 0
    last_capture_ts = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        detections = detector.detect(frame)
        face = largest_face(detections)

        if face is not None:
            draw_face_annotations(frame, [face])

        draw_app_chrome(
            frame,
            "TP Integrador - Registro",
            f"Persona: {current_name} | Capturas: {count_current} | Camara: {camera_index}",
            status,
            [
                ("ESPACIO", "capturar"),
                ("N", "otra persona"),
                ("T", "entrenar"),
                ("C", "camara"),
                ("Q", "salir"),
            ],
        )
        cv2.imshow("TP Integrador - Registro", frame)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()
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
            continue
        if key in (ord("n"), ord("N")):
            next_name = input("Nombre y apellido de la nueva persona: ").strip()
            if next_name:
                current_name = next_name
                count_current = 0
                status = f"Registrando: {current_name}"
            continue
        if key in (ord("t"), ord("T")):
            status = train_and_save()
            print(status)
            continue
        if key == 32 and (now - last_capture_ts) > 0.8:
            last_capture_ts = now
            if face is None:
                status = "No hay rostro detectado para capturar."
                continue
            aligned = align_face(frame, face)
            embedding = embedder.embed(aligned)
            path, photo_path = save_sample(current_name, embedding, aligned, save_photos)
            count_current += 1
            if photo_path is not None:
                status = f"Embedding y foto guardados: {path.name}"
                print(f"Foto guardada: {photo_path}")
            else:
                status = f"Embedding guardado: {path.name}"
            print(status)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

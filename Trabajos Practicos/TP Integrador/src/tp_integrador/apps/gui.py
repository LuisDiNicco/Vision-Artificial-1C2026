import argparse
from collections import deque
from pathlib import Path
import sys

from ..backend.logging_config import configure_native_logs


configure_native_logs()

import cv2
import numpy as np

from ..backend.alineamiento import align_face
from ..backend.camera import open_webcam
from ..backend.celebrity import CelebrityIndex, load_or_build_celebrity_cache
from ..backend.clasificador import Prediction, save_classifier, train_classifier, try_load_classifier
from ..backend.data import MODEL_PATH, count_embeddings_for_person, load_embeddings, save_sample
from ..backend.deteccion import MediaPipeFaceDetector, largest_face
from ..backend.embeddings import ArcFaceEmbedder
from ..backend.face_quality import assess_face_quality
from ..frontend.video_overlay import draw_face_annotations


VIDEO_W = 1280
VIDEO_H = 720
BASE_VIEWPORT_W = 1680
BASE_VIEWPORT_H = 900
SMOOTHING_WINDOW = 7
DETECTION_INTERVAL_REGISTRATION = 2
DETECTION_INTERVAL_RECOGNITION = 3
RECOGNITION_INTERVAL = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interfaz grafica del TP Integrador.")
    parser.add_argument("--camera", type=int, default=0, help="Indice de camara OpenCV a usar.")
    parser.add_argument("--max-camera-index", type=int, default=8, help="Indice maximo de camara a listar.")
    return parser.parse_args()


def main() -> None:
    enable_high_dpi()
    try:
        import dearpygui.dearpygui as dpg
    except ImportError as exc:
        raise RuntimeError("Falta Dear PyGui. Ejecuta: pip install dearpygui") from exc

    args = parse_args()
    app = FaceRecognitionGui(dpg, args.camera, args.max_camera_index)
    app.run()


class FaceRecognitionGui:
    def __init__(self, dpg, camera_index: int, max_camera_index: int) -> None:
        self.dpg = dpg
        self.camera_index = camera_index
        self.max_camera_index = max_camera_index
        self.mode = "registro"
        self.status = "Listo para iniciar"
        self.count_current = 0
        self.capture_requested = False
        self.train_requested = False
        self.celebrity_search_requested = False
        self.celebrity_cache_requested = False
        self.celebrity_matches = []
        self.recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
        self.frame_index = 0
        self.last_detections = []
        self.last_predictions = []
        self.video_display_w = VIDEO_W
        self.video_display_h = VIDEO_H

        self.cap = None
        self.detector = MediaPipeFaceDetector(max_faces=4)
        self.embedder = None
        self.classifier = None
        self.celebrity_index = None

    def run(self) -> None:
        self._build_ui()
        self._open_camera(self.camera_index)
        self._load_classifier_if_exists()

        while self.dpg.is_dearpygui_running():
            self._render_frame()
            self.dpg.render_dearpygui_frame()

        self._close()

    def _build_ui(self) -> None:
        dpg = self.dpg
        dpg.create_context()
        self._setup_fonts()

        with dpg.texture_registry(show=False):
            blank = np.zeros((VIDEO_H, VIDEO_W, 4), dtype=np.float32)
            dpg.add_dynamic_texture(VIDEO_W, VIDEO_H, blank.ravel(), tag="video_texture")

        with dpg.window(tag="main_window", label="TP Integrador - Reconocimiento Facial", no_close=True):
            with dpg.group(horizontal=True):
                with dpg.child_window(tag="sidebar_panel", width=340, height=820, border=True):
                    dpg.add_text("TP Integrador", tag="title_text", color=(245, 247, 250))
                    dpg.add_text("Reconocimiento facial", tag="subtitle_text", color=(120, 210, 255))
                    dpg.add_separator()

                    dpg.add_text("Modo")
                    dpg.add_radio_button(
                        ("Registro", "Reconocimiento"),
                        default_value="Registro",
                        callback=self._on_mode_changed,
                        horizontal=True,
                        tag="mode_radio",
                    )

                    dpg.add_spacer(height=10)
                    dpg.add_text("Persona")
                    dpg.add_input_text(
                        tag="person_name",
                        hint="Nombre y apellido",
                        width=-1,
                        callback=self._on_person_changed,
                    )
                    dpg.add_text("Embeddings guardados: 0", tag="person_count_text", color=(160, 210, 180))
                    dpg.add_checkbox(label="Guardar fotos alineadas", tag="save_photos", default_value=False)
                    dpg.add_checkbox(label="Espejar video", tag="mirror_video", default_value=False)

                    dpg.add_spacer(height=10)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Capturar", tag="capture_button", callback=self._request_capture, width=150)
                        dpg.add_button(label="Entrenar SVM", tag="train_button", callback=self._request_train, width=150)
                    dpg.add_button(label="Nueva persona", tag="new_person_button", callback=self._new_person, width=-1)

                    dpg.add_separator()
                    dpg.add_text("Doble famoso")
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Buscar doble", tag="celebrity_button", callback=self._request_celebrity_search, width=150)
                        dpg.add_button(label="Cachear famosos", tag="celebrity_cache_button", callback=self._request_celebrity_cache, width=150)
                    dpg.add_text("Cache: no cargado", tag="celebrity_status_text", wrap=300, color=(190, 200, 210))

                    dpg.add_separator()
                    dpg.add_text("Camara")
                    dpg.add_combo([], tag="camera_combo", width=-1)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Cambiar camara", tag="switch_camera_button", callback=self._switch_camera, width=150)
                        dpg.add_button(label="Siguiente", tag="next_camera_button", callback=self._next_camera, width=150)

                    dpg.add_separator()
                    dpg.add_text("Estado")
                    dpg.add_text("", tag="status_text", wrap=300, color=(225, 230, 235))
                    dpg.add_text("", tag="stats_text", wrap=300, color=(160, 170, 180))

                with dpg.child_window(tag="video_panel", width=1308, height=820, border=False):
                    dpg.add_image("video_texture", tag="video_image", width=VIDEO_W, height=VIDEO_H)

        self._apply_theme()
        self._populate_camera_options()
        dpg.create_viewport(title="TP Integrador - Reconocimiento Facial", width=BASE_VIEWPORT_W, height=BASE_VIEWPORT_H)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        dpg.set_viewport_resize_callback(self._on_viewport_resize)
        self._layout_to_viewport()

    def _setup_fonts(self) -> None:
        font_path = find_system_font()
        if font_path is None:
            return
        with self.dpg.font_registry():
            default_font = self.dpg.add_font(str(font_path), 18)
        self.dpg.bind_font(default_font)

    def _apply_theme(self) -> None:
        dpg = self.dpg
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (20, 23, 28))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (28, 32, 38))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (35, 132, 190))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (54, 160, 220))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (18, 104, 160))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (38, 43, 51))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (50, 57, 66))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (238, 242, 246))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (80, 210, 160))
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 14)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 9, 8)
        dpg.bind_theme(theme)

    def _render_frame(self) -> None:
        dpg = self.dpg
        if self.cap is None or not self.cap.isOpened():
            self._set_status("No hay camara abierta.")
            return

        ok, frame = self.cap.read()
        if not ok:
            self._set_status("No se pudo leer frame de camara.")
            return

        self.frame_index += 1
        if self.dpg.get_value("mirror_video"):
            frame = cv2.flip(frame, 1)
        detections = self._get_detections(frame)
        predictions = []

        if self.mode == "registro":
            face = largest_face(detections)
            if face is not None:
                draw_face_annotations(frame, [face])
            if self.capture_requested:
                self.capture_requested = False
                self._capture_sample(frame, face)
            if self.train_requested:
                self.train_requested = False
                self._train_model()
        else:
            if self.classifier is None:
                self._load_classifier_if_exists()
            if self.classifier is not None:
                predictions = self._get_predictions(frame, detections)
                draw_face_annotations(frame, detections, predictions)
            if self.celebrity_search_requested:
                self.celebrity_search_requested = False
                self._search_celebrity_double(frame, detections)

        if self.celebrity_cache_requested:
            self.celebrity_cache_requested = False
            self._build_celebrity_cache()

        if self.celebrity_matches:
            frame = self._compose_celebrity_view(frame)

        self._update_video_texture(frame)
        self._update_status_text(len(detections), predictions)

    def _capture_sample(self, frame, face) -> None:
        name = self.dpg.get_value("person_name").strip()
        if not name:
            self._set_status("Ingresa nombre y apellido antes de capturar.")
            return
        if face is None:
            self._set_status("No hay rostro detectado para capturar.")
            return
        quality = assess_face_quality(frame, face)
        if not quality.ok:
            self._set_status(f"No guardo la muestra: {quality.reason}")
            return

        aligned = align_face(frame, face)
        embedding = self._embedder().embed(aligned)
        _, photo_path = save_sample(name, embedding, aligned, self.dpg.get_value("save_photos"))
        self.count_current += 1
        self._update_person_count()
        suffix = " con foto" if photo_path is not None else ""
        self._set_status(f"Muestra guardada{suffix}. Total de esta sesion: {self.count_current}.")

    def _get_detections(self, frame):
        interval = DETECTION_INTERVAL_REGISTRATION if self.mode == "registro" else DETECTION_INTERVAL_RECOGNITION
        if self.capture_requested or self.frame_index % interval == 0 or not self.last_detections:
            self.last_detections = self.detector.detect(frame)
        return self.last_detections

    def _get_predictions(self, frame, detections):
        if self.frame_index % RECOGNITION_INTERVAL != 0 and self.last_predictions:
            return self.last_predictions

        predictions = []
        for detection in detections:
            quality = assess_face_quality(frame, detection)
            if not quality.ok:
                predictions.append(Prediction("desconocido", 0.0, float("inf"), method="calidad"))
                continue
            aligned = align_face(frame, detection)
            embedding = self._embedder().embed(aligned)
            predictions.append(self.classifier.predict(embedding))

        if len(predictions) == 1:
            self.recent_predictions.append(predictions[0])
            predictions[0] = smooth_single_prediction(self.recent_predictions)
        else:
            self.recent_predictions.clear()

        self.last_predictions = predictions
        return predictions

    def _train_model(self) -> None:
        embeddings, labels = load_embeddings()
        if len(labels) < 2:
            self._set_status("Necesitas al menos 2 capturas para entrenar.")
            return
        classifier = train_classifier(embeddings, labels)
        save_classifier(classifier, MODEL_PATH)
        self.classifier = classifier
        people = ", ".join(sorted(set(labels)))
        self._set_status(f"Modelo entrenado con {len(labels)} muestras: {people}")

    def _embedder(self):
        if self.embedder is None:
            self._set_status("Cargando extractor de embeddings...")
            self.embedder = ArcFaceEmbedder()
        return self.embedder

    def _load_classifier_if_exists(self) -> None:
        classifier, error = try_load_classifier(MODEL_PATH)
        if classifier is not None:
            self.classifier = classifier
            self._set_status("Modelo cargado.")
        else:
            self.classifier = None
            self._set_status(error)

    def _open_camera(self, index: int) -> None:
        next_cap = open_webcam(index, width=VIDEO_W, height=VIDEO_H, fps=30)
        if next_cap.isOpened():
            ok, _ = next_cap.read()
        else:
            ok = False

        if ok:
            if self.cap is not None:
                self.cap.release()
            self.cap = next_cap
            self.camera_index = index
            self.last_detections = []
            self.last_predictions = []
            self.dpg.set_value("camera_combo", str(index))
            self._set_status(f"Camara activa: {index}")
        else:
            next_cap.release()
            self.dpg.set_value("camera_combo", str(self.camera_index))
            self._set_status(f"No se pudo abrir camara {index}")

    def _populate_camera_options(self) -> None:
        items = [str(index) for index in range(self.max_camera_index + 1)]
        self.dpg.configure_item("camera_combo", items=items)
        self.dpg.set_value("camera_combo", str(self.camera_index))

    def _switch_camera(self, *args) -> None:
        selected = self.dpg.get_value("camera_combo")
        if selected == "":
            return
        self._open_camera(int(selected))

    def _next_camera(self, *args) -> None:
        start_index = self.camera_index
        for offset in range(1, self.max_camera_index + 2):
            next_index = (start_index + offset) % (self.max_camera_index + 1)
            before = self.camera_index
            self._open_camera(next_index)
            if self.camera_index != before:
                return
        self._set_status("No se encontro otra camara disponible.")

    def _on_mode_changed(self, sender, app_data, user_data=None) -> None:
        self.mode = "registro" if app_data == "Registro" else "reconocimiento"
        self.recent_predictions.clear()
        self._set_status(f"Modo activo: {app_data}")

    def _on_person_changed(self, sender, app_data, user_data=None) -> None:
        self._update_person_count()

    def _request_capture(self, *args) -> None:
        self.capture_requested = True

    def _request_train(self, *args) -> None:
        self.train_requested = True

    def _new_person(self, *args) -> None:
        self.dpg.set_value("person_name", "")
        self.count_current = 0
        self._update_person_count()
        self._set_status("Ingresa nombre y apellido de la nueva persona.")

    def _request_celebrity_search(self, *args) -> None:
        self.celebrity_search_requested = True

    def _request_celebrity_cache(self, *args) -> None:
        self.celebrity_cache_requested = True

    def _search_celebrity_double(self, frame, detections) -> None:
        face = largest_face(detections)
        if face is None:
            self._set_status("No hay rostro para buscar doble famoso.")
            return
        quality = assess_face_quality(frame, face)
        if not quality.ok:
            self._set_status(f"No busco doble famoso: {quality.reason}")
            return
        index = self._load_celebrity_index()
        if index is None:
            self._set_status("Primero cachea los embeddings de famosos.")
            return
        aligned = align_face(frame, face)
        embedding = self._embedder().embed(aligned)
        self.celebrity_matches = index.top_unique(embedding, limit=5)
        if self.celebrity_matches:
            names = ", ".join(match.name for match in self.celebrity_matches)
            self._set_status(f"Top 5 doble famoso: {names}")
        else:
            self._set_status("El cache de famosos esta vacio.")

    def _load_celebrity_index(self):
        if self.celebrity_index is not None:
            return self.celebrity_index
        if not CelebrityIndex.exists():
            self._update_celebrity_status("Cache: falta generar")
            return None
        self.celebrity_index = CelebrityIndex.load()
        self._update_celebrity_status(f"Cache: {len(self.celebrity_index.names)} embeddings")
        return self.celebrity_index

    def _build_celebrity_cache(self) -> None:
        self._set_status("Generando cache de famosos. Puede tardar varios minutos...")
        self._update_celebrity_status("Cache: generando...")
        self.celebrity_index = load_or_build_celebrity_cache(self._embedder(), self.detector)
        self._update_celebrity_status(f"Cache: {len(self.celebrity_index.names)} embeddings")
        self._set_status("Cache de famosos listo.")

    def _update_celebrity_status(self, message: str) -> None:
        if self.dpg.does_item_exist("celebrity_status_text"):
            self.dpg.set_value("celebrity_status_text", message)

    def _compose_celebrity_view(self, frame):
        canvas = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        left_w = int(VIDEO_W * 0.58)
        right_w = VIDEO_W - left_w
        webcam = cv2.resize(frame, (left_w, VIDEO_H), interpolation=cv2.INTER_AREA)
        canvas[:, :left_w] = webcam
        canvas[:, left_w:] = (24, 27, 32)
        cv2.line(canvas, (left_w, 0), (left_w, VIDEO_H), (70, 180, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Top 5 doble famoso", (left_w + 18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 240, 245), 2, cv2.LINE_AA)

        tile_h = 126
        y = 58
        for rank, match in enumerate(self.celebrity_matches[:5], start=1):
            photo = cv2.imread(str(match.image_path))
            if photo is None:
                photo = np.full((112, 112, 3), 42, dtype=np.uint8)
            photo = cv2.resize(photo, (112, 112), interpolation=cv2.INTER_AREA)
            x = left_w + 18
            canvas[y:y + 112, x:x + 112] = photo
            text_x = x + 128
            cv2.putText(canvas, f"{rank}. {match.name}"[:34], (text_x, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 247, 250), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"sim {match.similarity * 100:.1f}%  dist {match.distance:.3f}", (text_x, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (145, 210, 235), 1, cv2.LINE_AA)
            y += tile_h
        return canvas

    def _set_status(self, message: str) -> None:
        self.status = message
        if self.dpg.does_item_exist("status_text"):
            self.dpg.set_value("status_text", message)

    def _update_status_text(self, face_count: int, predictions) -> None:
        if not self.dpg.does_item_exist("stats_text"):
            return
        prediction_text = ""
        if predictions:
            prediction = predictions[0]
            prediction_text = f" | {prediction.label}: {prediction.confidence * 100:.1f}%"
        self.dpg.set_value(
            "stats_text",
            f"Modo: {self.mode} | Camara: {self.camera_index} | Rostros: {face_count}{prediction_text}",
        )

    def _update_person_count(self) -> None:
        if not self.dpg.does_item_exist("person_count_text"):
            return
        name = self.dpg.get_value("person_name").strip()
        count = count_embeddings_for_person(name) if name else 0
        self.dpg.set_value("person_count_text", f"Embeddings guardados: {count}")

    def _update_video_texture(self, frame_bgr) -> None:
        frame = cv2.resize(frame_bgr, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        data = (frame.astype(np.float32) / 255.0).ravel()
        self.dpg.set_value("video_texture", data)

    def _on_viewport_resize(self, *args) -> None:
        self._layout_to_viewport()

    def _layout_to_viewport(self) -> None:
        width = max(self.dpg.get_viewport_client_width(), 900)
        height = max(self.dpg.get_viewport_client_height(), 620)
        margin = 18
        gap = 14
        scale = float(np.clip(width / BASE_VIEWPORT_W, 0.90, 1.35))
        self.dpg.set_global_font_scale(scale)

        sidebar_w = int(np.clip(width * 0.19, 320, 460))
        content_w = max(width - sidebar_w - (margin * 2) - gap, 480)
        content_h = max(height - (margin * 2), 360)
        video_w = content_w
        video_h = int(video_w * 9 / 16)
        if video_h > content_h:
            video_h = content_h
            video_w = int(video_h * 16 / 9)

        self.video_display_w = video_w
        self.video_display_h = video_h
        if self.dpg.does_item_exist("sidebar_panel"):
            self.dpg.configure_item("sidebar_panel", width=sidebar_w, height=content_h)
            button_w = max(120, int((sidebar_w - 34) / 2))
            self.dpg.configure_item("capture_button", width=button_w)
            self.dpg.configure_item("train_button", width=button_w)
            self.dpg.configure_item("switch_camera_button", width=button_w)
            self.dpg.configure_item("next_camera_button", width=button_w)
            self.dpg.configure_item("celebrity_button", width=button_w)
            self.dpg.configure_item("celebrity_cache_button", width=button_w)
        if self.dpg.does_item_exist("video_panel"):
            self.dpg.configure_item("video_panel", width=content_w, height=content_h)
            self.dpg.configure_item("video_image", width=video_w, height=video_h)

    def _close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.detector.close()
        self.dpg.destroy_context()


def smooth_single_prediction(recent_predictions):
    grouped = {}
    for prediction in recent_predictions:
        grouped.setdefault(prediction.label, []).append(prediction)
    label = max(grouped, key=lambda key: (len(grouped[key]), sum(item.confidence for item in grouped[key])))
    group = grouped[label]
    confidence = sum(item.confidence for item in group) / len(group)
    distance = sum(item.distance for item in group) / len(group)
    return type(group[-1])(label=label, confidence=confidence, distance=distance, method=group[-1].method)


def enable_high_dpi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        # Per-monitor DPI awareness v2. Avoids Windows bitmap-scaling the whole GUI.
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass


def find_system_font() -> Path | None:
    candidates = []
    if sys.platform == "win32":
        candidates.extend(
            [
                Path("C:/Windows/Fonts/segoeui.ttf"),
                Path("C:/Windows/Fonts/arial.ttf"),
            ]
        )
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


if __name__ == "__main__":
    main()

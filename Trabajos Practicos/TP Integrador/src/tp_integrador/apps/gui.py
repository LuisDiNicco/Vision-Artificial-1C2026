import argparse
from collections import deque
from pathlib import Path
import sys
import threading
import time

from ..backend.logging_config import configure_native_logs


configure_native_logs()

import cv2
import numpy as np

from ..backend.camera import open_webcam
from ..backend.celebrity import (
    CELEBRITY_MIN_MARGIN,
    CelebrityIndex,
    celebrity_match_rejection_reason,
    load_or_build_celebrity_cache,
)
from ..backend.clasificador import Prediction, save_classifier, train_classifier, try_load_classifier
from ..backend.data import MODEL_PATH, count_embeddings_for_person, load_embeddings, save_sample
from ..backend.deteccion import MediaPipeFaceDetector, largest_face
from ..backend.embeddings import ArcFaceEmbedder
from ..backend.face_quality import VIDEO_FACE_QUALITY, assess_face_quality
from ..backend.video_analysis_cache import (
    analysis_at_time,
    analysis_cache_path,
    load_video_analysis,
    make_analysis_record,
    prepare_analysis_timeline,
    save_video_analysis,
)
from ..backend.video_inputs import VIDEO_DOWNLOAD_DIR, download_youtube_video, looks_like_youtube_url
from ..frontend.gui.help import HELP_TOPICS
from ..frontend.gui.layout import build_main_window, build_support_windows
from ..frontend.native_file_dialog import choose_video_file
from ..frontend.video_overlay import draw_face_annotations


VIDEO_W = 1280
VIDEO_H = 720
VIDEO_TEXTURE_TIERS = ((960, 540), (1280, 720), (1920, 1080))
BASE_VIEWPORT_W = 1680
BASE_VIEWPORT_H = 900
SMOOTHING_WINDOW = 7
EMBEDDING_SMOOTHING_WINDOW = 5
DETECTION_INTERVAL_REGISTRATION = 1
DETECTION_INTERVAL_RECOGNITION = 1
RECOGNITION_INTERVAL = 3
REGISTRATION_MIN_QUALITY_SCORE = 0.62
VIDEO_UNCERTAIN_GRACE_SAMPLES = 2
VIDEO_MISSING_FACE_GRACE_FRAMES = 10
VIDEO_FACE_CONTINUITY_MIN_IOU = 0.18
VIDEO_DETECTION_PERIOD_SECONDS = 0.10
VIDEO_PREPROCESS_MAX_FACES = 12


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
    def __init__(
        self,
        dpg,
        camera_index: int,
        max_camera_index: int,
    ) -> None:
        self.dpg = dpg
        self.camera_index = camera_index
        self.max_camera_index = max_camera_index
        self.mode = "registro"
        self.status = "Listo para iniciar"
        self.count_current = 0
        self.capture_requested = False
        self.train_requested = False
        self.video_file_path = ""
        self.youtube_video_path = None
        self.video_actor_results = []
        self.video_playback_cap = None
        self.video_playback_path = None
        self.video_playback_frame_index = 0
        self.video_playback_last_predictions = []
        self.video_playback_last_detections = []
        self.video_playback_last_detection_seconds = float("-inf")
        self.video_playback_detection_interval = 1
        self.video_playback_recognition_interval = 8
        self.video_playback_min_similarity = 0.34
        self.video_playback_frame_duration = 1.0 / 25.0
        self.video_playback_last_frame_time = 0.0
        self.video_playback_paused = False
        self.video_playback_seek_pending = False
        self.video_playback_fps = 25.0
        self.video_playback_duration = 0.0
        self.video_playback_clock_base = 0.0
        self.video_playback_clock_started_at = 0.0
        self.video_playback_displayed_frame = -1
        self.video_playback_current_seconds = 0.0
        self.video_playback_recognition_period = 0.35
        self.video_playback_last_recognition_seconds = float("-inf")
        self.video_playback_lock = threading.RLock()
        self.video_recognition_lock = threading.RLock()
        self.video_recent_embeddings = deque(maxlen=EMBEDDING_SMOOTHING_WINDOW)
        self.video_recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
        self.video_uncertain_samples = 0
        self.video_missing_face_frames = 0
        self.video_last_face_bbox = None
        self.static_display_frame = None
        self.recent_embeddings = deque(maxlen=EMBEDDING_SMOOTHING_WINDOW)
        self.recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
        self.frame_index = 0
        self.last_detections = []
        self.last_predictions = []
        self.video_display_w = VIDEO_W
        self.video_display_h = VIDEO_H
        self.video_source_w = VIDEO_W
        self.video_source_h = VIDEO_H
        self.video_texture_w = VIDEO_W
        self.video_texture_h = VIDEO_H
        self.video_texture_tag = video_texture_tag(VIDEO_W, VIDEO_H)
        self.video_controls_visible = False
        self.video_playback_uses_cache = False
        self.video_cached_times = []
        self.video_cached_records = []
        self.video_preprocess_lock = threading.RLock()
        self.video_preprocess_active = False
        self.video_preprocess_progress = 0.0
        self.video_preprocess_overlay = "Esperando..."
        self.video_preprocess_result = None
        self.video_preprocess_error = None
        self.video_preprocess_finished_pending = False
        self.video_preprocess_cancel = threading.Event()
        self.video_preprocess_thread = None

        self.cap = None
        self.detector = MediaPipeFaceDetector(max_faces=4, min_detection_confidence=0.70, min_tracking_confidence=0.70)
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

        with dpg.texture_registry(show=False, tag="video_texture_registry"):
            blank = np.zeros((VIDEO_H, VIDEO_W, 4), dtype=np.float32)
            dpg.add_dynamic_texture(VIDEO_W, VIDEO_H, blank.ravel(), tag=self.video_texture_tag)

        build_main_window(dpg, self, VIDEO_W, VIDEO_H, self.video_texture_tag)
        build_support_windows(dpg, self)

        self._apply_theme()
        self._populate_camera_options()
        dpg.create_viewport(title="TP Integrador - Reconocimiento Facial", width=BASE_VIEWPORT_W, height=BASE_VIEWPORT_H)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        dpg.set_viewport_resize_callback(self._on_viewport_resize)
        self._layout_to_viewport()

    def _show_help(self, topic: str) -> None:
        title, body = HELP_TOPICS.get(topic, HELP_TOPICS["general"])
        self.dpg.set_value("help_title", title)
        self.dpg.set_value("help_body", body)
        self.dpg.configure_item("help_modal", show=True)

    def _hide_help(self, *args) -> None:
        self.dpg.configure_item("help_modal", show=False)

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
        if self._poll_video_preprocessing():
            return
        if self.video_playback_cap is not None:
            self._render_actor_video_frame()
            return

        if self.static_display_frame is not None:
            self._update_video_texture(self.static_display_frame)
            return

        if self.cap is None or not self.cap.isOpened():
            self._set_status("No hay fuente de video abierta.")
            return

        ok, frame = self.cap.read()
        if not ok:
            self._set_status("No se pudo leer frame de la fuente de video.")
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
        if quality.score < REGISTRATION_MIN_QUALITY_SCORE:
            self._set_status(f"No guardo la muestra: calidad {quality.score:.2f}; busca mejor luz, enfoque y pose frontal.")
            return

        embedding, aligned = self._embedder().embed_face(frame, face)
        metadata = {
            "quality_score": quality.score,
            "quality_reason": quality.reason,
            "bbox": face.bbox,
            "detection_confidence": face.confidence,
            "embedder_backend": self._embedder().backend_name,
            "alignment_backend": self._embedder().alignment_backend,
        }
        _, photo_path = save_sample(name, embedding, aligned, self.dpg.get_value("save_photos"), metadata)
        self.count_current += 1
        self._update_person_count()
        suffix = " con foto" if photo_path is not None else ""
        self._set_status(f"Muestra guardada{suffix}. Calidad {quality.score:.2f}. Total de esta sesion: {self.count_current}.")

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
                if len(detections) == 1:
                    self.recent_embeddings.clear()
                predictions.append(Prediction("desconocido", 0.0, float("inf"), method="calidad"))
                continue
            embedding, _ = self._embedder().embed_face(frame, detection)
            if len(detections) == 1:
                self.recent_embeddings.append(embedding)
                embedding = average_embeddings(self.recent_embeddings)
            predictions.append(self.classifier.predict(embedding))

        if len(predictions) == 1:
            self.recent_predictions.append(predictions[0])
            predictions[0] = smooth_single_prediction(self.recent_predictions)
            if len(self.recent_embeddings) > 1 and predictions[0].label != "desconocido":
                predictions[0].method = f"{predictions[0].method}+promedio"
        else:
            self.recent_embeddings.clear()
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
        outliers = getattr(classifier, "outlier_sample_count", 0)
        used = getattr(classifier, "training_sample_count", len(labels))
        suffix = f" ({outliers} outliers excluidos)" if outliers else ""
        self._set_status(f"Modelo entrenado con {used}/{len(labels)} muestras{suffix}: {people}")

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

    def _reset_video_state(self) -> None:
        self._stop_actor_video()
        self.last_detections = []
        self.last_predictions = []
        self.recent_embeddings.clear()
        self.recent_predictions.clear()
        self.video_actor_results = []
        self.static_display_frame = None

    def _open_camera(self, index: int) -> bool:
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
            self._reset_video_state()
            self.dpg.set_value("camera_combo", str(index))
            self._set_status(f"Camara activa: {index}")
            return True
        else:
            next_cap.release()
            self.dpg.set_value("camera_combo", str(self.camera_index))
            self._set_status(f"No se pudo abrir camara {index}")
            return False

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

    def _activate_training_mode(self, *args) -> None:
        self.mode = "registro"
        self.recent_embeddings.clear()
        self.recent_predictions.clear()
        self._set_status("Modo entrenamiento activo. Captura muestras de una persona por vez.")

    def _activate_recognition_mode(self, *args) -> None:
        self.mode = "reconocimiento"
        self.recent_embeddings.clear()
        self.recent_predictions.clear()
        self._set_status("Modo reconocimiento activo.")

    def _on_mode_changed(self, sender, app_data, user_data=None) -> None:
        self.mode = "registro" if app_data == "Registro" else "reconocimiento"
        self.recent_embeddings.clear()
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

    def _show_video_file_dialog(self, *args) -> None:
        try:
            VIDEO_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
            initial_path = self.video_file_path or str(VIDEO_DOWNLOAD_DIR)
            path = choose_video_file(initial_path)
        except Exception as exc:
            self._set_status(f"No se pudo abrir el explorador de archivos: {exc}")
            return
        if path is None:
            return
        self.video_file_path = str(path)
        self.youtube_video_path = None
        if self.dpg.does_item_exist("youtube_url_input"):
            self.dpg.set_value("youtube_url_input", "")
        display_name = Path(path).name
        self.dpg.set_value("video_file_text", display_name)
        self._set_status(f"Video seleccionado: {display_name}")

    def _preprocess_selected_video(self, *args) -> None:
        with self.video_preprocess_lock:
            if self.video_preprocess_active:
                self._set_status("Ya hay un video en proceso de analisis.")
                return
        video_path = self._resolve_actor_video_path()
        if video_path is None:
            return
        if self._load_celebrity_index() is None:
            self._set_status("Primero cachea los embeddings de famosos.")
            return

        sample_seconds = float(self.dpg.get_value("video_sample_seconds"))
        min_similarity = float(self.dpg.get_value("video_min_similarity"))
        cache_path = analysis_cache_path(Path(video_path), sample_seconds, min_similarity)
        if cache_path.exists():
            self._set_status("El analisis de este video ya esta guardado en cache.")
            return

        self._stop_actor_video()
        self.video_preprocess_cancel.clear()
        with self.video_preprocess_lock:
            self.video_preprocess_active = True
            self.video_preprocess_progress = 0.0
            self.video_preprocess_overlay = "Preparando modelos..."
            self.video_preprocess_result = None
            self.video_preprocess_error = None
            self.video_preprocess_finished_pending = False
        self._set_preprocess_controls(False)
        self.dpg.configure_item("video_preprocess_progress", show=True)
        self.dpg.set_value("video_preprocess_progress", 0.0)
        self.dpg.configure_item("video_preprocess_progress", overlay="Preparando modelos...")
        self._set_status("Preprocesando video. La reproduccion quedara disponible al finalizar.")

        thread = threading.Thread(
            target=self._preprocess_video_worker,
            args=(Path(video_path), cache_path, sample_seconds, min_similarity),
            name="video-preprocessing",
            daemon=True,
        )
        self.video_preprocess_thread = thread
        thread.start()

    def _preprocess_video_worker(
        self,
        video_path: Path,
        cache_path: Path,
        sample_seconds: float,
        min_similarity: float,
    ) -> None:
        cap = None
        detector = None
        started_at = time.monotonic()
        try:
            if self.embedder is None:
                self.embedder = ArcFaceEmbedder()
            detector = MediaPipeFaceDetector(
                max_faces=VIDEO_PREPROCESS_MAX_FACES,
                min_detection_confidence=0.70,
                min_tracking_confidence=0.70,
            )
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError("No se pudo abrir el video para preprocesarlo.")

            fps = max(float(cap.get(cv2.CAP_PROP_FPS) or 25.0), 1.0)
            total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
            if total_frames <= 0:
                raise RuntimeError("El video no informa una cantidad valida de frames.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps
            # El preprocesamiento asincronico prioriza precision: analiza cada
            # frame. El muestreo configurado se conserva para el modo en vivo.
            targets = range(total_frames)
            sample_count = total_frames

            self.video_playback_min_similarity = min_similarity
            self.video_playback_recognition_period = sample_seconds
            self.video_playback_last_recognition_seconds = float("-inf")
            self.video_playback_last_predictions = []
            self._reset_video_recognition_history()
            records = []

            for sample_index, target_frame in enumerate(targets):
                if self.video_preprocess_cancel.is_set():
                    raise RuntimeError("Analisis cancelado.")
                decoder_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                for _ in range(max(0, target_frame - decoder_frame)):
                    if not cap.grab():
                        break
                ok, frame = cap.read()
                if not ok:
                    break

                seconds = target_frame / fps
                self.video_playback_current_seconds = seconds
                detections = detector.detect(frame)
                predictions = self._get_celebrity_video_predictions(
                    frame,
                    detections,
                    seconds,
                    force_recognition=True,
                )
                records.append(make_analysis_record(seconds, detections, predictions))

                progress = min(1.0, (sample_index + 1) / sample_count)
                elapsed = time.monotonic() - started_at
                remaining = elapsed * (1.0 - progress) / progress if progress > 0 else 0.0
                overlay = f"{progress * 100:.0f}% | restante aprox. {format_video_time(remaining)}"
                with self.video_preprocess_lock:
                    self.video_preprocess_progress = progress
                    self.video_preprocess_overlay = overlay

            save_video_analysis(
                cache_path,
                video_path,
                duration,
                fps,
                width,
                height,
                sample_seconds,
                min_similarity,
                records,
            )
            result = cache_path
            error = None
        except Exception as exc:
            result = None
            error = str(exc)
        finally:
            if cap is not None:
                cap.release()
            if detector is not None:
                detector.close()
            self._reset_video_recognition_history()
            with self.video_preprocess_lock:
                self.video_preprocess_active = False
                self.video_preprocess_result = result
                self.video_preprocess_error = error
                self.video_preprocess_finished_pending = True

    def _poll_video_preprocessing(self) -> bool:
        with self.video_preprocess_lock:
            active = self.video_preprocess_active
            progress = self.video_preprocess_progress
            overlay = self.video_preprocess_overlay
            finished = self.video_preprocess_finished_pending
            result = self.video_preprocess_result
            error = self.video_preprocess_error
            if finished:
                self.video_preprocess_finished_pending = False

        if self.dpg.does_item_exist("video_preprocess_progress") and (active or finished):
            self.dpg.configure_item("video_preprocess_progress", show=True, overlay=overlay)
            self.dpg.set_value("video_preprocess_progress", progress)
        if finished:
            self._set_preprocess_controls(True)
            if error:
                self._set_status(f"No se pudo preprocesar el video: {error}")
            else:
                self.dpg.set_value("video_preprocess_progress", 1.0)
                self.dpg.configure_item("video_preprocess_progress", overlay="Analisis completo - guardado en cache")
                self._set_status(f"Analisis terminado y guardado: {Path(result).name}")
        return active

    def _set_preprocess_controls(self, enabled: bool) -> None:
        for tag in (
            "choose_video_button",
            "analyze_video_button",
            "preprocess_video_button",
            "live_video_button",
        ):
            if self.dpg.does_item_exist(tag):
                self.dpg.configure_item(tag, enabled=enabled)

    def _on_workflow_tab_changed(self, sender, app_data, user_data=None) -> None:
        video_tab_id = self.dpg.get_alias_id("video_workflow_tab")
        was_visible = self.video_controls_visible
        self.video_controls_visible = app_data in {video_tab_id, "video_workflow_tab"}
        if was_visible and not self.video_controls_visible and self.video_playback_cap is not None:
            self._stop_actor_video()
            self.static_display_frame = None
            self._set_status("Vista en vivo restaurada.")
        if self.dpg.does_item_exist("video_playback_controls"):
            self.dpg.configure_item("video_playback_controls", show=self.video_controls_visible)
        self._layout_to_viewport()

    def _analyze_selected_video(self, *args) -> None:
        self._start_selected_video()

    def _start_selected_video(self) -> None:
        video_path = self._resolve_actor_video_path()
        if video_path is None:
            return
        sample_seconds = float(self.dpg.get_value("video_sample_seconds"))
        min_similarity = float(self.dpg.get_value("video_min_similarity"))
        cached_payload = None
        cache_path = analysis_cache_path(Path(video_path), sample_seconds, min_similarity)
        if cache_path.exists():
            try:
                cached_payload = load_video_analysis(cache_path)
            except (OSError, ValueError):
                cached_payload = None
        if cached_payload is None:
            index = self._load_celebrity_index()
            if index is None:
                self._set_status("Primero cachea los embeddings de famosos.")
                return

        next_cap = cv2.VideoCapture(str(video_path))
        if not next_cap.isOpened():
            next_cap.release()
            self._set_status(f"No se pudo abrir el video: {video_path}")
            return
        self._stop_actor_video()
        with self.video_playback_lock:
            self.video_playback_cap = next_cap
            self.video_playback_path = str(video_path)
            self.video_playback_frame_index = 0
            self.video_playback_last_predictions = []
            self.video_playback_last_detections = []
            self.video_playback_last_detection_seconds = float("-inf")
            self.video_playback_min_similarity = min_similarity
            fps = next_cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.video_playback_fps = max(float(fps), 1.0)
            frame_count = max(float(next_cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0.0)
            self.video_playback_duration = frame_count / self.video_playback_fps if frame_count else 0.0
            self.video_playback_frame_duration = 1.0 / self.video_playback_fps
            self.video_playback_last_frame_time = 0.0
            self.video_playback_paused = False
            self.video_playback_seek_pending = False
            self.video_playback_clock_base = 0.0
            self.video_playback_clock_started_at = time.monotonic()
            self.video_playback_displayed_frame = -1
            self.video_playback_current_seconds = 0.0
            self.video_playback_recognition_period = sample_seconds
            self.video_playback_last_recognition_seconds = float("-inf")
            self.video_playback_recognition_interval = max(1, int(round(fps * sample_seconds)))
            self.video_playback_uses_cache = cached_payload is not None
            if cached_payload is not None:
                self.video_cached_times, self.video_cached_records = prepare_analysis_timeline(cached_payload)
            else:
                self.video_cached_times = []
                self.video_cached_records = []
        self._reset_video_recognition_history()
        self.static_display_frame = None
        if cached_payload is None:
            self._embedder()
        with self.video_playback_lock:
            self.video_playback_clock_started_at = time.monotonic()
        self._configure_playback_controls(active=True)
        self._set_status("Reproduciendo video con reconocimiento de famosos.")

    def _return_to_live_video(self, *args) -> None:
        self._stop_actor_video()
        self.static_display_frame = None
        self._set_status("Vista en vivo restaurada.")

    def _stop_actor_video(self) -> None:
        with self.video_playback_lock:
            if self.video_playback_cap is not None:
                self.video_playback_cap.release()
            self.video_playback_cap = None
            self.video_playback_path = None
            self.video_playback_last_predictions = []
            self.video_playback_last_detections = []
            self.video_playback_last_detection_seconds = float("-inf")
            self.video_playback_frame_index = 0
            self.video_playback_last_frame_time = 0.0
            self.video_playback_paused = False
            self.video_playback_seek_pending = False
            self.video_playback_duration = 0.0
            self.video_playback_clock_base = 0.0
            self.video_playback_clock_started_at = 0.0
            self.video_playback_displayed_frame = -1
            self.video_playback_current_seconds = 0.0
            self.video_playback_last_recognition_seconds = float("-inf")
            self.video_playback_uses_cache = False
            self.video_cached_times = []
            self.video_cached_records = []
        self._reset_video_recognition_history()
        self._configure_playback_controls(active=False)

    def _toggle_actor_video_playback(self, *args) -> None:
        with self.video_playback_lock:
            if self.video_playback_cap is None:
                return
            now = time.monotonic()
            if self.video_playback_paused:
                self.video_playback_clock_started_at = now
                self.video_playback_paused = False
            else:
                self.video_playback_clock_base = self._playback_position_locked(now)
                self.video_playback_paused = True
            paused = self.video_playback_paused
            self.video_playback_last_frame_time = now
        self.dpg.configure_item(
            "video_play_pause_button",
            label=">" if paused else "||",
        )
        state = "en pausa" if paused else "reanudado"
        self._set_status(f"Video {state}.")

    def _seek_actor_video(self, sender, app_data, user_data=None) -> None:
        self._seek_actor_video_to(float(app_data))

    def _skip_actor_video(self, sender, app_data, user_data=None) -> None:
        offset_seconds = float(user_data or 0.0)
        with self.video_playback_lock:
            if self.video_playback_cap is None:
                return
            current_seconds = self._playback_position_locked()
        self._seek_actor_video_to(current_seconds + offset_seconds)

    def _seek_actor_video_to(self, requested_seconds: float) -> None:
        with self.video_playback_lock:
            if self.video_playback_cap is None:
                return
            seconds = max(0.0, min(requested_seconds, self.video_playback_duration))
            target_frame = max(0, int(round(seconds * self.video_playback_fps)))
            self.video_playback_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.video_playback_frame_index = target_frame
            self.video_playback_displayed_frame = target_frame - 1
            self.video_playback_current_seconds = seconds
            self.video_playback_clock_base = seconds
            self.video_playback_clock_started_at = time.monotonic()
            self.video_playback_last_recognition_seconds = float("-inf")
            self.video_playback_last_predictions = []
            self.video_playback_last_detections = []
            self.video_playback_last_detection_seconds = float("-inf")
            self.video_playback_last_frame_time = 0.0
            self.video_playback_seek_pending = True
        self._reset_video_recognition_history()
        self._update_playback_time(seconds)

    def _playback_position_locked(self, now: float | None = None) -> float:
        if self.video_playback_paused:
            position = self.video_playback_clock_base
        else:
            now = time.monotonic() if now is None else now
            position = self.video_playback_clock_base + max(0.0, now - self.video_playback_clock_started_at)
        if self.video_playback_duration:
            position = min(position, self.video_playback_duration)
        return max(0.0, position)

    def _configure_playback_controls(self, active: bool) -> None:
        if not self.dpg.does_item_exist("video_play_pause_button"):
            return
        self.dpg.configure_item("video_play_pause_button", enabled=active, label="||" if active else ">")
        self.dpg.configure_item("video_skip_back_button", enabled=active)
        self.dpg.configure_item("video_skip_forward_button", enabled=active)
        self.dpg.configure_item("video_seek_slider", enabled=active)
        self.dpg.configure_item(
            "video_seek_slider",
            min_value=0.0,
            max_value=max(self.video_playback_duration, 1.0),
        )
        self.dpg.set_value("video_seek_slider", 0.0)
        self._update_playback_time(0.0)

    def _update_playback_time(self, seconds: float) -> None:
        if self.dpg.does_item_exist("video_playback_time"):
            self.dpg.set_value(
                "video_playback_time",
                f"{format_video_time(seconds)} / {format_video_time(self.video_playback_duration)}",
            )

    def _on_video_landmarks_changed(self, *args) -> None:
        with self.video_playback_lock:
            if self.video_playback_cap is not None:
                self.video_playback_seek_pending = True

    def _render_actor_video_frame(self) -> None:
        with self.video_playback_lock:
            if self.video_playback_cap is None:
                return
            if self.video_playback_paused and not self.video_playback_seek_pending:
                return
            current_seconds = self._playback_position_locked()
            if self.video_playback_duration and current_seconds >= self.video_playback_duration:
                reached_end = True
            else:
                reached_end = False
            target_frame = max(0, int(current_seconds * self.video_playback_fps))
            if not self.video_playback_seek_pending and target_frame <= self.video_playback_displayed_frame:
                return
            if reached_end:
                ok = False
                frame = None
            else:
                decoder_frame = int(self.video_playback_cap.get(cv2.CAP_PROP_POS_FRAMES))
                frames_behind = target_frame - decoder_frame
                if frames_behind > 2:
                    if frames_behind <= int(self.video_playback_fps):
                        for _ in range(frames_behind):
                            if not self.video_playback_cap.grab():
                                break
                    else:
                        self.video_playback_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                elif frames_behind < -1:
                    self.video_playback_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ok, frame = self.video_playback_cap.read()
                actual_frame = max(0, int(self.video_playback_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                self.video_playback_frame_index = actual_frame
                self.video_playback_displayed_frame = actual_frame
                self.video_playback_current_seconds = current_seconds
            self.video_playback_seek_pending = False
        if not ok:
            self._stop_actor_video()
            self._set_status("Fin del video. Vista en vivo restaurada.")
            return

        self.dpg.set_value("video_seek_slider", current_seconds)
        self._update_playback_time(current_seconds)
        if self.video_playback_uses_cache:
            detections, predictions = analysis_at_time(
                self.video_cached_times,
                self.video_cached_records,
                current_seconds,
            )
        else:
            detections = self._get_video_detections(frame, current_seconds)
            predictions = self._get_celebrity_video_predictions(frame, detections, current_seconds)
        show_landmarks = (
            bool(self.dpg.get_value("video_show_landmarks"))
            if self.dpg.does_item_exist("video_show_landmarks")
            else True
        )
        draw_face_annotations(
            frame,
            detections,
            predictions,
            show_landmarks=show_landmarks,
        )
        self._update_video_texture(frame)
        self._update_actor_video_stats(len(detections), predictions)

    def _get_video_detections(self, frame, current_seconds: float):
        if current_seconds - self.video_playback_last_detection_seconds < VIDEO_DETECTION_PERIOD_SECONDS:
            return self.video_playback_last_detections
        self.video_playback_last_detection_seconds = current_seconds
        self.video_playback_last_detections = self.detector.detect(frame)
        return self.video_playback_last_detections

    def _get_celebrity_video_predictions(
        self,
        frame,
        detections,
        current_seconds: float | None = None,
        force_recognition: bool = False,
    ):
        with self.video_recognition_lock:
            current_seconds = self.video_playback_current_seconds if current_seconds is None else current_seconds
            if not detections:
                self.video_missing_face_frames += 1
                if self.video_missing_face_frames > VIDEO_MISSING_FACE_GRACE_FRAMES:
                    self._reset_video_recognition_history()
                    self.video_playback_last_predictions = []
                return []

            self.video_missing_face_frames = 0
            if len(detections) == 1:
                current_bbox = detections[0].bbox
                if (
                    self.video_last_face_bbox is not None
                    and bbox_iou(self.video_last_face_bbox, current_bbox) < VIDEO_FACE_CONTINUITY_MIN_IOU
                ):
                    self._reset_video_recognition_history()
                    self.video_playback_last_predictions = []
                self.video_last_face_bbox = current_bbox
            else:
                self.video_last_face_bbox = None

            if (
                not force_recognition
                and current_seconds - self.video_playback_last_recognition_seconds
                < self.video_playback_recognition_period
                and len(self.video_playback_last_predictions) == len(detections)
            ):
                return self.video_playback_last_predictions
            self.video_playback_last_recognition_seconds = current_seconds

            index = self._load_celebrity_index()
            predictions = []
            for detection in detections:
                quality = assess_face_quality(frame, detection, VIDEO_FACE_QUALITY)
                if not quality.ok:
                    if len(detections) == 1:
                        self.video_recent_embeddings.clear()
                        uncertain = self._video_uncertain_prediction("calidad")
                        if uncertain.label != "desconocido":
                            self.video_playback_last_predictions = [uncertain]
                            return [uncertain]
                    predictions.append(Prediction("desconocido", 0.0, float("inf"), method="calidad"))
                    continue

                embedding, _ = self._embedder().embed_face(frame, detection)
                if len(detections) == 1:
                    self.video_recent_embeddings.append(embedding)
                    embedding = average_embeddings(self.video_recent_embeddings)

                matches = index.top_unique(embedding, limit=2) if index is not None else []
                rejection_reason = celebrity_match_rejection_reason(
                    matches,
                    self.video_playback_min_similarity,
                    CELEBRITY_MIN_MARGIN,
                )
                if rejection_reason is not None:
                    if len(detections) == 1:
                        uncertain = self._video_uncertain_prediction(rejection_reason)
                        if uncertain.label != "desconocido":
                            self.video_playback_last_predictions = [uncertain]
                            return [uncertain]
                    predictions.append(Prediction("desconocido", 0.0, float("inf"), method="famosos"))
                    continue
                match = matches[0]
                self.video_uncertain_samples = 0
                confidence = float(np.clip((match.similarity - self.video_playback_min_similarity) / 0.30, 0.0, 1.0))
                predictions.append(Prediction(match.name, confidence, match.distance, method="famosos"))

            if len(predictions) == 1:
                self.video_recent_predictions.append(predictions[0])
                predictions[0] = smooth_single_prediction(self.video_recent_predictions)
                if len(self.video_recent_embeddings) > 1 and predictions[0].label != "desconocido":
                    predictions[0].method = f"{predictions[0].method}+promedio"
            else:
                self._reset_video_recognition_history()

            self.video_playback_last_predictions = predictions
            return predictions

    def _reset_video_recognition_history(self) -> None:
        with self.video_recognition_lock:
            self.video_recent_embeddings.clear()
            self.video_recent_predictions.clear()
            self.video_uncertain_samples = 0
            self.video_missing_face_frames = 0
            self.video_last_face_bbox = None

    def _video_uncertain_prediction(self, reason: str) -> Prediction:
        self.video_uncertain_samples += 1
        if (
            self.video_uncertain_samples <= VIDEO_UNCERTAIN_GRACE_SAMPLES
            and len(self.video_playback_last_predictions) == 1
            and self.video_playback_last_predictions[0].label != "desconocido"
        ):
            previous = self.video_playback_last_predictions[0]
            return Prediction(
                previous.label,
                previous.confidence * 0.90,
                previous.distance,
                method=f"{reason}+histeresis",
            )

        self.video_recent_embeddings.clear()
        self.video_recent_predictions.clear()
        return Prediction("desconocido", 0.0, float("inf"), method=reason)

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

    def _resolve_actor_video_path(self) -> Path | str | None:
        youtube_url = self.dpg.get_value("youtube_url_input").strip() if self.dpg.does_item_exist("youtube_url_input") else ""
        if youtube_url:
            if not looks_like_youtube_url(youtube_url):
                self._set_status("La URL no parece ser de YouTube.")
                return None
            try:
                self.youtube_video_path = download_youtube_video(youtube_url, progress_callback=self._set_status)
            except Exception as exc:
                self._set_status(f"No se pudo descargar el video de YouTube: {exc}")
                return None
            self.video_file_path = str(self.youtube_video_path)
            self.dpg.set_value("video_file_text", f"YouTube: {Path(self.video_file_path).name}")
            return self.youtube_video_path
        if not self.video_file_path:
            self._set_status("Elegi un video local o pega una URL de YouTube.")
            return None
        return self.video_file_path

    def _update_video_actor_results(self) -> None:
        if not self.dpg.does_item_exist("video_results_text"):
            return
        if not self.video_actor_results:
            self.dpg.set_value("video_results_text", "Sin coincidencias confiables.")
            return
        lines = []
        for result in self.video_actor_results[:8]:
            lines.append(
                f"{result.name}: conf {result.confidence * 100:.0f}% | "
                f"sim {result.similarity * 100:.1f}% | "
                f"{result.samples} muestras | {result.first_second:.1f}s-{result.last_second:.1f}s"
            )
        self.dpg.set_value("video_results_text", "\n".join(lines))

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
            f"Modo: {self.mode} | Fuente: Camara {self.camera_index} | Rostros: {face_count}{prediction_text}",
        )

    def _update_actor_video_stats(self, face_count: int, predictions) -> None:
        if not self.dpg.does_item_exist("stats_text"):
            return
        prediction_text = ""
        if predictions:
            known = [prediction for prediction in predictions if prediction.label != "desconocido"]
            if known:
                names = ", ".join(prediction.label for prediction in known[:3])
                prediction_text = f" | Detectados: {names}"
            else:
                prediction_text = " | Detectados: desconocido"
        path = Path(self.video_playback_path).name if self.video_playback_path else "video"
        self.dpg.set_value(
            "stats_text",
            f"Modo: video | Fuente: {path} | Rostros: {face_count}{prediction_text}",
        )
        if self.dpg.does_item_exist("video_results_text"):
            if not predictions:
                self.dpg.set_value("video_results_text", "Sin rostros detectados en este frame.")
            else:
                lines = [
                    f"{prediction.label}: {prediction.confidence * 100:.0f}%"
                    for prediction in predictions[:6]
                ]
                self.dpg.set_value("video_results_text", "\n".join(lines))

    def _update_person_count(self) -> None:
        if not self.dpg.does_item_exist("person_count_text"):
            return
        name = self.dpg.get_value("person_name").strip()
        count = count_embeddings_for_person(name) if name else 0
        self.dpg.set_value("person_count_text", f"Embeddings guardados: {count}")

    def _update_video_texture(self, frame_bgr) -> None:
        source_h, source_w = frame_bgr.shape[:2]
        self.video_source_w = source_w
        self.video_source_h = source_h
        self._ensure_source_appropriate_texture()
        texture_w = self.video_texture_w
        texture_h = self.video_texture_h
        scale = min(texture_w / max(source_w, 1), texture_h / max(source_h, 1))
        target_w = max(1, int(round(source_w * scale)))
        target_h = max(1, int(round(source_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=interpolation)
        frame = np.zeros((texture_h, texture_w, 3), dtype=np.uint8)
        x = (texture_w - target_w) // 2
        y = (texture_h - target_h) // 2
        frame[y : y + target_h, x : x + target_w] = resized
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        data = (frame.astype(np.float32) / 255.0).ravel()
        self.dpg.set_value(self.video_texture_tag, data)

    def _ensure_video_texture(self, display_w: int, display_h: int) -> None:
        texture_w, texture_h = select_video_texture_size(display_w, display_h)
        if (texture_w, texture_h) == (self.video_texture_w, self.video_texture_h):
            return

        next_tag = video_texture_tag(texture_w, texture_h)
        if not self.dpg.does_item_exist(next_tag):
            blank = np.zeros((texture_h, texture_w, 4), dtype=np.float32)
            self.dpg.add_dynamic_texture(
                texture_w,
                texture_h,
                blank.ravel(),
                tag=next_tag,
                parent="video_texture_registry",
            )
        self.dpg.configure_item("video_image", texture_tag=next_tag)
        self.video_texture_w = texture_w
        self.video_texture_h = texture_h
        self.video_texture_tag = next_tag

    def _ensure_source_appropriate_texture(self) -> None:
        required_w = min(self.video_display_w, max(self.video_source_w, VIDEO_TEXTURE_TIERS[0][0]))
        required_h = min(self.video_display_h, max(self.video_source_h, VIDEO_TEXTURE_TIERS[0][1]))
        self._ensure_video_texture(required_w, required_h)

    def _on_viewport_resize(self, *args) -> None:
        self._layout_to_viewport()

    def _layout_to_viewport(self) -> None:
        width = max(self.dpg.get_viewport_client_width(), 900)
        height = max(self.dpg.get_viewport_client_height(), 620)
        margin = 18
        gap = 14
        scale = float(np.clip(width / BASE_VIEWPORT_W, 0.90, 1.35))
        self.dpg.set_global_font_scale(scale)

        sidebar_w = int(np.clip(width * 0.24, 360, 520))
        content_w = max(width - sidebar_w - (margin * 2) - gap, 480)
        content_h = max(height - (margin * 2), 360)
        controls_h = 72 if self.video_controls_visible else 0
        available_video_h = max(content_h - controls_h, 240)
        video_w = content_w
        video_h = int(video_w * 9 / 16)
        if video_h > available_video_h:
            video_h = available_video_h
            video_w = int(video_h * 16 / 9)

        self.video_display_w = video_w
        self.video_display_h = video_h
        self._ensure_source_appropriate_texture()
        if self.dpg.does_item_exist("sidebar_panel"):
            self.dpg.configure_item("sidebar_panel", width=sidebar_w, height=content_h)
            button_w = max(120, int((sidebar_w - 34) / 2))
            self.dpg.configure_item("training_mode_button", width=button_w)
            self.dpg.configure_item("capture_button", width=button_w)
            self.dpg.configure_item("train_button", width=button_w)
            self.dpg.configure_item("new_person_button", width=button_w)
            self.dpg.configure_item("recognition_mode_button", width=-1)
            self.dpg.configure_item("switch_camera_button", width=button_w)
            self.dpg.configure_item("next_camera_button", width=button_w)
            self.dpg.configure_item("choose_video_button", width=-1)
            self.dpg.configure_item("analyze_video_button", width=-1)
            self.dpg.configure_item("preprocess_video_button", width=-1)
            self.dpg.configure_item("live_video_button", width=-1)
        if self.dpg.does_item_exist("video_panel"):
            self.dpg.configure_item("video_panel", width=content_w, height=content_h)
            self.dpg.configure_item("video_image", width=video_w, height=video_h)
            self.dpg.configure_item("video_seek_slider", width=max(video_w, 240))

    def _close(self) -> None:
        self.video_preprocess_cancel.set()
        if self.video_preprocess_thread is not None and self.video_preprocess_thread.is_alive():
            self.video_preprocess_thread.join(timeout=2.0)
        self._stop_actor_video()
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


def average_embeddings(recent_embeddings):
    matrix = np.vstack(list(recent_embeddings)).astype(np.float32)
    embedding = matrix.mean(axis=0)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype(np.float32)


def format_video_time(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def select_video_texture_size(display_w: int, display_h: int) -> tuple[int, int]:
    for width, height in VIDEO_TEXTURE_TIERS:
        if display_w <= width and display_h <= height:
            return width, height
    return VIDEO_TEXTURE_TIERS[-1]


def video_texture_tag(width: int, height: int) -> str:
    return f"video_texture_{width}x{height}"


def bbox_iou(first, second) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


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

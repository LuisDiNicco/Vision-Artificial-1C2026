"""Perfil reproducible del preprocesamiento offline sobre varios frames.

No altera el pipeline: envuelve puntos de medicion y escribe el JSON de salida
fuera del cache normal. Ejecutar desde la raiz de TP Integrador.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import MethodType

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tp_integrador.backend import embeddings as embeddings_module  # noqa: E402
from tp_integrador.backend import offline_video_analysis as offline  # noqa: E402
from tp_integrador.backend.celebrity import CelebrityIndex  # noqa: E402
from tp_integrador.backend.embeddings import ArcFaceEmbedder  # noqa: E402
from tp_integrador.backend.video_analysis_cache import save_video_analysis  # noqa: E402


class Timings:
    def __init__(self) -> None:
        self.seconds = Counter()
        self.calls = Counter()

    def add(self, name: str, elapsed: float) -> None:
        self.seconds[name] += elapsed
        self.calls[name] += 1

    def wrap(self, owner, name: str, label: str | None = None) -> None:
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.add(label or name, time.perf_counter() - start)

        setattr(owner, name, measured)


class ProfiledCapture:
    original = cv2.VideoCapture
    timings: Timings

    def __init__(self, *args, **kwargs):
        self._capture = self.original(*args, **kwargs)

    def read(self):
        start = time.perf_counter()
        try:
            return self._capture.read()
        finally:
            self.timings.add("decoding", time.perf_counter() - start)

    def __getattr__(self, name):
        return getattr(self._capture, name)


class ProfiledDetector:
    original = offline.MediaPipeFaceDetector
    timings: Timings
    instances = 0

    def __init__(self, *args, **kwargs):
        self._detector = self.original(*args, **kwargs)
        self._label = "primary_detection" if self.instances % 2 == 0 else "multiscale_detection"
        type(self).instances += 1

    def detect(self, frame):
        start = time.perf_counter()
        try:
            return self._detector.detect(frame)
        finally:
            self.timings.add(self._label, time.perf_counter() - start)

    def close(self):
        return self._detector.close()


class ResourceSampler:
    def __init__(self) -> None:
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        process = psutil.Process()

        def sample():
            process.cpu_percent(None)
            while not self._stop.wait(0.5):
                self.samples.append((process.cpu_percent(None), process.memory_info().rss))

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def finish(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            return {}
        return {
            "process_cpu_percent_mean": sum(x[0] for x in self.samples) / len(self.samples),
            "process_cpu_percent_peak": max(x[0] for x in self.samples),
            "rss_mib_peak": max(x[1] for x in self.samples) / 1024**2,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--secondary", choices=("always", "adaptive"), default="always")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    timings = Timings()
    ProfiledCapture.timings = timings
    ProfiledDetector.timings = timings
    cv2.VideoCapture = ProfiledCapture
    offline.MediaPipeFaceDetector = ProfiledDetector

    for name, label in (
        ("_nms_detections", "nms"),
        ("_assign_tracks", "tracking"),
        ("_refine_unresolved_tracks", "second_pass"),
    ):
        timings.wrap(offline, name, label)

    original_align = embeddings_module.align_face

    def measured_align(*align_args, **align_kwargs):
        start = time.perf_counter()
        try:
            return original_align(*align_args, **align_kwargs)
        finally:
            timings.add("alignment", time.perf_counter() - start)

    embeddings_module.align_face = measured_align

    index = CelebrityIndex.load()
    timings.wrap(index, "top_unique", "celebrity_comparison")
    timings.wrap(index, "sample_embeddings_for_person", "celebrity_comparison")
    embedder = ArcFaceEmbedder()
    original_embed = embedder.embed
    inside_embed_face = False

    def measured_embed(self, image):
        label = "arcface_normal" if inside_embed_face else "arcface_flip"
        start = time.perf_counter()
        try:
            return original_embed(image)
        finally:
            timings.add(label, time.perf_counter() - start)

    embedder.embed = MethodType(measured_embed, embedder)
    original_embed_face = embedder.embed_face

    def measured_embed_face(self, frame, detection):
        nonlocal inside_embed_face
        inside_embed_face = True
        try:
            return original_embed_face(frame, detection)
        finally:
            inside_embed_face = False

    embedder.embed_face = MethodType(measured_embed_face, embedder)
    original_tta_batch = embedder.embed_tta_batch

    def measured_tta_batch(self, images):
        start = time.perf_counter()
        try:
            return original_tta_batch(images)
        finally:
            timings.add("arcface_tta_batch", time.perf_counter() - start)

    embedder.embed_tta_batch = MethodType(measured_tta_batch, embedder)

    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    result = offline.analyze_video_offline(
        args.video,
        embedder,
        index,
        config=offline.OfflineVideoConfig(
            max_frames=args.frames,
            embedding_batch_size=args.batch_size,
            secondary_strategy=args.secondary,
            parallel_pipeline=args.parallel,
        ),
    )
    analysis_elapsed = time.perf_counter() - started

    fd, raw_temp = tempfile.mkstemp(prefix="tp-video-profile-", suffix=".json")
    os.close(fd)
    temp_path = Path(raw_temp)
    try:
        started = time.perf_counter()
        save_video_analysis(
            temp_path,
            args.video,
            result.duration,
            result.fps,
            result.width,
            result.height,
            0.0,
            0.34,
            result.records,
            result.metadata,
        )
        timings.add("json_serialization", time.perf_counter() - started)
        json_bytes = temp_path.stat().st_size
    finally:
        temp_path.unlink(missing_ok=True)

    resources = sampler.finish()
    processed = int(result.metadata["frames_analyzed"])
    report = {
        "video": str(args.video),
        "backend": embedder.backend_name,
        "alignment_backend": embedder.alignment_backend,
        "embedding_batch_size": args.batch_size,
        "secondary_strategy": args.secondary,
        "parallel_pipeline": args.parallel,
        "frames": processed,
        "source_fps": result.fps,
        "source_seconds": processed / result.fps,
        "wall_seconds_analysis": analysis_elapsed,
        "wall_seconds_with_json": analysis_elapsed + timings.seconds["json_serialization"],
        "frames_per_second": processed / max(analysis_elapsed, 1e-9),
        "faces": int(result.metadata["detections"]),
        "faces_per_second": int(result.metadata["detections"]) / max(analysis_elapsed, 1e-9),
        "two_minute_extrapolation_seconds": analysis_elapsed * 120.0 / max(processed / result.fps, 1e-9),
        "json_bytes": json_bytes,
        "timings_seconds": dict(timings.seconds),
        "timing_calls": dict(timings.calls),
        "metadata": result.metadata,
        "resources": resources,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

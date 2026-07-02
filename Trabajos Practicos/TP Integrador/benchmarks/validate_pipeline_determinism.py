"""Comprueba que el doble buffer no altera records, caras, tracks ni etiquetas."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tp_integrador.backend.celebrity import CelebrityIndex  # noqa: E402
from tp_integrador.backend.embeddings import ArcFaceEmbedder  # noqa: E402
from tp_integrador.backend.offline_video_analysis import (  # noqa: E402
    OfflineVideoConfig,
    analyze_video_offline,
)


def main() -> None:
    video = ROOT / "cache" / "videos" / "Video TP Vision.mp4"
    embedder = ArcFaceEmbedder()
    index = CelebrityIndex.load()
    started = time.perf_counter()
    sequential = analyze_video_offline(
        video,
        embedder,
        index,
        config=OfflineVideoConfig(max_frames=120, parallel_pipeline=False),
    )
    sequential_seconds = time.perf_counter() - started
    started = time.perf_counter()
    parallel = analyze_video_offline(
        video,
        embedder,
        index,
        config=OfflineVideoConfig(max_frames=120, parallel_pipeline=True),
    )
    parallel_seconds = time.perf_counter() - started
    confidences = [
        float(face["confidence"])
        for record in parallel.records
        for face in record.get("faces", [])
        if face.get("label") != "desconocido"
    ]
    print(
        json.dumps(
            {
                "records_equal": sequential.records == parallel.records,
                "frames": len(sequential.records),
                "sequential_seconds": sequential_seconds,
                "parallel_seconds": parallel_seconds,
                "confidence_min": min(confidences) if confidences else None,
                "confidence_max": max(confidences) if confidences else None,
                "confidence_unique_rounded": len({round(value, 4) for value in confidences}),
                "sequential_metadata": sequential.metadata,
                "parallel_metadata": parallel.metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

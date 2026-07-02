"""Ejecuta el flujo productivo completo, guarda features/decisiones y mide reuso."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tp_integrador.backend.celebrity import CelebrityIndex  # noqa: E402
from tp_integrador.backend.embeddings import ArcFaceEmbedder  # noqa: E402
from tp_integrador.backend.offline_video_analysis import analyze_video_offline  # noqa: E402
from tp_integrador.backend.video_analysis_cache import (  # noqa: E402
    analysis_cache_path,
    save_video_analysis,
    video_feature_cache_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--min-similarity", type=float, default=0.34)
    parser.add_argument("--sample-seconds", type=float, default=0.33)
    args = parser.parse_args()
    embedder = ArcFaceEmbedder()
    index = CelebrityIndex.load()
    feature_path = video_feature_cache_path(args.video)
    decision_path = analysis_cache_path(args.video, args.sample_seconds, args.min_similarity)

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    result = analyze_video_offline(
        args.video,
        embedder,
        index,
        min_similarity=args.min_similarity,
        feature_cache_path=feature_path,
    )
    analysis_seconds = time.perf_counter() - wall_started
    process_cpu_seconds = time.process_time() - cpu_started
    serialize_started = time.perf_counter()
    save_video_analysis(
        decision_path,
        args.video,
        result.duration,
        result.fps,
        result.width,
        result.height,
        args.sample_seconds,
        args.min_similarity,
        result.records,
        result.metadata,
    )
    serialization_seconds = time.perf_counter() - serialize_started
    print(
        json.dumps(
            {
                "video": str(args.video),
                "feature_cache": str(feature_path),
                "feature_cache_mib": feature_path.stat().st_size / 1024**2,
                "decision_cache": str(decision_path),
                "decision_cache_mib": decision_path.stat().st_size / 1024**2,
                "analysis_seconds": analysis_seconds,
                "json_serialization_seconds": serialization_seconds,
                "process_cpu_seconds": process_cpu_seconds,
                "logical_cpu_equivalents": process_cpu_seconds / max(analysis_seconds, 1e-9),
                "frames_per_second": result.metadata["frames_analyzed"] / analysis_seconds,
                "faces_per_second": result.metadata["detections"] / analysis_seconds,
                "metadata": result.metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

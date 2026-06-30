"""Compara inferencia ArcFace individual vs batch sobre conocidos y desconocidos."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tp_integrador.backend.celebrity import (  # noqa: E402
    CELEBRITY_MIN_MARGIN,
    CelebrityIndex,
    celebrity_match_rejection_reason,
)
from tp_integrador.backend.embeddings import ArcFaceEmbedder  # noqa: E402


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def existing_path(path: Path) -> Path | None:
    candidates = [path, ROOT / path, ROOT / "cache" / "famosos" / "images" / path.name]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def select_images(index: CelebrityIndex) -> list[tuple[str, Path]]:
    by_name = defaultdict(list)
    for name, path in zip(index.sample_names, index.sample_image_paths):
        resolved = existing_path(path)
        if resolved is not None:
            by_name[name].append(resolved)

    selected = []
    for name in sorted(by_name)[:4]:
        selected.extend((name, path) for path in by_name[name][:2])

    unknowns = sorted((ROOT / "datos_privados" / "fotos").glob("*/*"))[:3]
    selected.extend(("__unknown__", path) for path in unknowns if path.is_file())
    return selected


def main() -> None:
    index = CelebrityIndex.load()
    samples = select_images(index)
    images = []
    labels = []
    paths = []
    for label, path in samples:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            labels.append(label)
            paths.append(path)
            images.append(image)
    if len(images) < 4:
        raise RuntimeError("No hay suficientes imagenes locales para validar el batch.")

    embedder = ArcFaceEmbedder()
    sequential = [
        normalized(embedder.embed(image) + embedder.embed(cv2.flip(image, 1)))
        for image in images
    ]
    batched = embedder.embed_tta_batch(images)

    rows = []
    for label, path, before, after in zip(labels, paths, sequential, batched):
        difference = np.abs(before - after)
        before_matches = index.top_unique(before, limit=5)
        after_matches = index.top_unique(after, limit=5)
        top_before = [match.name for match in before_matches]
        top_after = [match.name for match in after_matches]
        rows.append(
            {
                "expected_group": label,
                "file": str(path.relative_to(ROOT)),
                "cosine": float(before @ after),
                "max_abs_difference": float(difference.max()),
                "mean_abs_difference": float(difference.mean()),
                "top5_equal": top_before == top_after,
                "top5_before": top_before,
                "top5_batch": top_after,
                "top_similarity": float(after_matches[0].similarity),
                "top_margin": float(after_matches[0].similarity - after_matches[1].similarity),
                "rejected": celebrity_match_rejection_reason(
                    after_matches,
                    0.34,
                    CELEBRITY_MIN_MARGIN,
                )
                is not None,
            }
        )
    print(
        json.dumps(
            {
                "images": len(rows),
                "people": len({label for label in labels if label != "__unknown__"}),
                "unknown_images": labels.count("__unknown__"),
                "minimum_cosine": min(row["cosine"] for row in rows),
                "maximum_abs_difference": max(row["max_abs_difference"] for row in rows),
                "mean_abs_difference": float(
                    np.mean([row["mean_abs_difference"] for row in rows])
                ),
                "all_top5_equal": all(row["top5_equal"] for row in rows),
                "samples": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

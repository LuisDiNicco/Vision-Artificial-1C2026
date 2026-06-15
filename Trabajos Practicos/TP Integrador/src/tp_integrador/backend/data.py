import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[3]
PRIVATE_DATA_DIR = BASE_DIR / "datos_privados"
EMBEDDINGS_DIR = PRIVATE_DATA_DIR / "embeddings"
PHOTOS_DIR = PRIVATE_DATA_DIR / "fotos"
MODEL_PATH = BASE_DIR / "modelo" / "clasificador_svm.joblib"
SUMMARY_PATH = BASE_DIR / "datos" / "resumen_embeddings.json"


def person_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "persona"


def save_sample(
    name: str,
    embedding: np.ndarray,
    aligned_face_bgr: np.ndarray,
    save_photo: bool,
) -> Tuple[Path, Optional[Path]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slug = person_slug(name)
    person_dir = EMBEDDINGS_DIR / slug
    person_dir.mkdir(parents=True, exist_ok=True)
    path = person_dir / f"{timestamp}.npz"
    np.savez_compressed(path, name=name.strip(), embedding=embedding.astype(np.float32))

    if save_photo:
        photo_dir = PHOTOS_DIR / slug
        photo_dir.mkdir(parents=True, exist_ok=True)
        photo_path = photo_dir / f"{timestamp}.jpg"
        ok, encoded = cv2.imencode(".jpg", aligned_face_bgr)
        if not ok:
            raise RuntimeError("No se pudo codificar la foto alineada como JPG.")
        photo_path.write_bytes(encoded.tobytes())
    else:
        photo_path = None

    write_public_summary()
    return path, photo_path


def load_embeddings() -> Tuple[np.ndarray, List[str]]:
    embeddings = []
    labels: List[str] = []
    for path in sorted(EMBEDDINGS_DIR.glob("*/*.npz")):
        data = np.load(path, allow_pickle=False)
        embeddings.append(data["embedding"].astype(np.float32))
        labels.append(str(data["name"]))
    if not embeddings:
        return np.empty((0, 512), dtype=np.float32), []
    return np.vstack(embeddings), labels


def write_public_summary() -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for path in sorted(EMBEDDINGS_DIR.glob("*/*.npz")):
        data = np.load(path, allow_pickle=False)
        name = str(data["name"])
        summary[name] = summary.get(name, 0) + 1
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

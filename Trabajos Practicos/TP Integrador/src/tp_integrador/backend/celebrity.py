from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

from .data import BASE_DIR
from .deteccion import MediaPipeFaceDetector, largest_face
from .face_quality import assess_face_quality


CELEBRITY_DATASET = "ares1123/celebrity_dataset"
CELEBRITY_CACHE_DIR = BASE_DIR / "cache" / "famosos"
CELEBRITY_IMAGES_DIR = CELEBRITY_CACHE_DIR / "images"
CELEBRITY_INDEX_PATH = CELEBRITY_CACHE_DIR / "celebrity_arcface_index.npz"
CELEBRITY_MIN_MARGIN = 0.04


@dataclass
class CelebrityMatch:
    name: str
    similarity: float
    distance: float
    image_path: Path
    samples: int = 1


def celebrity_match_rejection_reason(
    matches: List[CelebrityMatch],
    min_similarity: float,
    min_margin: float = CELEBRITY_MIN_MARGIN,
) -> Optional[str]:
    if not matches or matches[0].similarity < min_similarity:
        return "umbral"
    if len(matches) > 1 and matches[0].similarity - matches[1].similarity + 1e-6 < min_margin:
        return "margen"
    return None


class CelebrityIndex:
    def __init__(
        self,
        embeddings: np.ndarray,
        names: List[str],
        image_paths: List[Path],
        counts: Optional[List[int]] = None,
        sample_embeddings: Optional[np.ndarray] = None,
        sample_names: Optional[List[str]] = None,
        sample_image_paths: Optional[List[Path]] = None,
        sample_person_ids: Optional[np.ndarray] = None,
    ) -> None:
        self.embeddings = embeddings.astype(np.float32)
        self.names = names
        self.image_paths = image_paths
        self.counts = counts or [1] * len(names)
        self.sample_embeddings = sample_embeddings
        self.sample_names = sample_names or []
        self.sample_image_paths = sample_image_paths or []
        self.sample_person_ids = sample_person_ids

    @classmethod
    def load(cls, path: Path = CELEBRITY_INDEX_PATH) -> "CelebrityIndex":
        data = np.load(path, allow_pickle=False)
        if "person_embeddings" in data:
            return cls(
                embeddings=data["person_embeddings"].astype(np.float32),
                names=[str(item) for item in data["person_names"]],
                image_paths=[Path(str(item)) for item in data["person_image_paths"]],
                counts=[int(item) for item in data["person_counts"]],
                sample_embeddings=data["embeddings"].astype(np.float32),
                sample_names=[str(item) for item in data["names"]],
                sample_image_paths=[Path(str(item)) for item in data["image_paths"]],
                sample_person_ids=data["sample_person_ids"].astype(np.int32)
                if "sample_person_ids" in data
                else None,
            )

        sample_embeddings = data["embeddings"].astype(np.float32)
        sample_names = [str(item) for item in data["names"]]
        sample_image_paths = [Path(str(item)) for item in data["image_paths"]]
        embeddings, names, image_paths, counts = _aggregate_people(
            sample_embeddings,
            sample_names,
            sample_image_paths,
        )
        return cls(
            embeddings=embeddings,
            names=names,
            image_paths=image_paths,
            counts=counts,
        )

    @staticmethod
    def exists(path: Path = CELEBRITY_INDEX_PATH) -> bool:
        return path.exists()

    def top_unique(self, embedding: np.ndarray, limit: int = 5) -> List[CelebrityMatch]:
        if self.embeddings.size == 0:
            return []
        query = embedding.astype(np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-6)
        similarities = self.embeddings @ query
        order = np.argsort(-similarities)
        matches: List[CelebrityMatch] = []
        seen = set()
        for idx in order:
            name = self.names[int(idx)]
            if name in seen:
                continue
            seen.add(name)
            similarity = float(similarities[int(idx)])
            distance = float(np.sqrt(max(0.0, 2.0 - 2.0 * similarity)))
            matches.append(
                CelebrityMatch(
                    name,
                    similarity,
                    distance,
                    self.image_paths[int(idx)],
                    self.counts[int(idx)],
                )
            )
            if len(matches) >= limit:
                break
        return matches

    def sample_embeddings_for_person(self, name: str) -> np.ndarray:
        if self.sample_embeddings is None:
            return np.empty((0, 512), dtype=np.float32)
        if self.sample_person_ids is not None and name in self.names:
            person_id = self.names.index(name)
            return self.sample_embeddings[self.sample_person_ids == person_id]
        mask = np.array(self.sample_names) == name
        return self.sample_embeddings[mask]


def save_celebrity_index(
    embeddings: List[np.ndarray],
    names: List[str],
    image_paths: List[Path],
    path: Path = CELEBRITY_INDEX_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.vstack(embeddings).astype(np.float32) if embeddings else np.empty((0, 512), dtype=np.float32)
    person_embeddings, person_names, person_image_paths, person_counts = _aggregate_people(matrix, names, image_paths)
    person_id_by_name = {name: idx for idx, name in enumerate(person_names)}
    sample_person_ids = np.array([person_id_by_name[name] for name in names], dtype=np.int32)
    np.savez_compressed(
        path,
        embeddings=matrix,
        names=np.array(names, dtype=str),
        image_paths=np.array([str(item) for item in image_paths], dtype=str),
        person_embeddings=person_embeddings,
        person_names=np.array(person_names, dtype=str),
        person_image_paths=np.array([str(item) for item in person_image_paths], dtype=str),
        person_counts=np.array(person_counts, dtype=np.int32),
        sample_person_ids=sample_person_ids,
    )


def load_or_build_celebrity_cache(
    embedder,
    detector: Optional[MediaPipeFaceDetector] = None,
    force: bool = False,
    limit: Optional[int] = None,
    max_per_person: Optional[int] = None,
) -> CelebrityIndex:
    if CELEBRITY_INDEX_PATH.exists() and not force:
        return CelebrityIndex.load()
    return build_celebrity_cache(
        embedder,
        detector=detector,
        limit=limit,
        max_per_person=max_per_person,
    )


def build_celebrity_cache(
    embedder,
    detector: Optional[MediaPipeFaceDetector] = None,
    limit: Optional[int] = None,
    max_per_person: Optional[int] = None,
    progress_every: int = 100,
) -> CelebrityIndex:
    from datasets import load_dataset

    owns_detector = detector is None
    detector = detector or MediaPipeFaceDetector(max_faces=1, min_detection_confidence=0.55)
    dataset = load_dataset(CELEBRITY_DATASET, split="train")
    label_names = _label_names(dataset)
    CELEBRITY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    embeddings: List[np.ndarray] = []
    names: List[str] = []
    image_paths: List[Path] = []
    per_person: dict[str, int] = {}

    try:
        for row in _limited(dataset, limit):
            name = _row_name(row, label_names)
            if max_per_person is not None and per_person.get(name, 0) >= max_per_person:
                continue

            image_bgr = _pil_to_bgr(row["image"])
            detections = detector.detect(image_bgr)
            face = largest_face(detections)
            if face is None:
                continue
            if not assess_face_quality(image_bgr, face).ok:
                continue

            embedding, aligned = embedder.embed_face(image_bgr, face)
            image_path = CELEBRITY_IMAGES_DIR / f"{len(embeddings):06d}.jpg"
            _write_jpg(image_path, aligned)

            embeddings.append(embedding)
            names.append(name)
            image_paths.append(image_path)
            per_person[name] = per_person.get(name, 0) + 1

            if progress_every and len(embeddings) % progress_every == 0:
                print(f"Cache famosos: {len(embeddings)} embeddings generados...")
    finally:
        if owns_detector:
            detector.close()

    save_celebrity_index(embeddings, names, image_paths)
    matrix = np.vstack(embeddings).astype(np.float32) if embeddings else np.empty((0, 512), dtype=np.float32)
    person_embeddings, person_names, person_image_paths, person_counts = _aggregate_people(matrix, names, image_paths)
    person_id_by_name = {name: idx for idx, name in enumerate(person_names)}
    sample_person_ids = np.array([person_id_by_name[name] for name in names], dtype=np.int32)
    return CelebrityIndex(
        person_embeddings,
        person_names,
        person_image_paths,
        person_counts,
        sample_embeddings=matrix,
        sample_names=names,
        sample_image_paths=image_paths,
        sample_person_ids=sample_person_ids,
    )


def _aggregate_people(
    embeddings: np.ndarray,
    names: List[str],
    image_paths: List[Path],
) -> tuple[np.ndarray, List[str], List[Path], List[int]]:
    if embeddings.size == 0:
        return np.empty((0, 512), dtype=np.float32), [], [], []

    person_embeddings = []
    person_names = []
    person_image_paths = []
    person_counts = []
    names_array = np.array(names)

    for name in sorted(set(names)):
        indices = np.where(names_array == name)[0]
        samples = embeddings[indices]
        centroid = samples.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-6)
        similarities = samples @ centroid
        representative_idx = int(indices[int(np.argmax(similarities))])

        person_embeddings.append(centroid.astype(np.float32))
        person_names.append(name)
        person_image_paths.append(image_paths[representative_idx])
        person_counts.append(int(len(indices)))

    return np.vstack(person_embeddings).astype(np.float32), person_names, person_image_paths, person_counts


def _limited(dataset, limit: Optional[int]) -> Iterable:
    count = 0
    for row in dataset:
        if limit is not None and count >= limit:
            break
        count += 1
        yield row


def _label_names(dataset) -> List[str]:
    feature = dataset.features.get("label")
    names = getattr(feature, "names", None)
    return list(names) if names else []


def _row_name(row, label_names: List[str]) -> str:
    label = row["label"]
    if isinstance(label, str):
        return label
    if label_names and int(label) < len(label_names):
        return label_names[int(label)]
    return str(label)


def _pil_to_bgr(image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _write_jpg(path: Path, image_bgr: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError(f"No se pudo codificar imagen JPG para {path}")
    path.write_bytes(encoded.tobytes())

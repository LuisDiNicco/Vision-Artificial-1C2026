from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


@dataclass
class Prediction:
    label: str
    confidence: float
    distance: float
    method: str = "distancia"


class FaceSVMClassifier:
    def __init__(
        self,
        pipeline: Optional[Pipeline],
        label_encoder: LabelEncoder,
        centroids: Dict[str, np.ndarray],
        distance_threshold: float,
        reference_embeddings: Optional[np.ndarray] = None,
        reference_labels: Optional[List[str]] = None,
        class_thresholds: Optional[Dict[str, float]] = None,
        class_core_distances: Optional[Dict[str, float]] = None,
        probability_threshold: float = 0.55,
    ) -> None:
        self.pipeline = pipeline
        self.label_encoder = label_encoder
        self.centroids = centroids
        self.distance_threshold = distance_threshold
        self.reference_embeddings = reference_embeddings
        self.reference_labels = reference_labels or []
        self.class_thresholds = class_thresholds or {}
        self.class_core_distances = class_core_distances or {}
        self.probability_threshold = probability_threshold

    def predict(self, embedding: np.ndarray) -> Prediction:
        self._ensure_runtime_defaults()
        if not self.centroids:
            return Prediction("desconocido", 0.0, float("inf"))

        nearest_label, nearest_distance = self._nearest_reference(embedding)
        calibrated_confidence = self._calibrated_confidence(nearest_label, nearest_distance)
        threshold = self.class_thresholds.get(nearest_label, self.distance_threshold)

        if self.pipeline is None:
            label = nearest_label if nearest_distance <= threshold else "desconocido"
            return Prediction(label, calibrated_confidence if label != "desconocido" else 0.0, nearest_distance)

        probabilities = self.pipeline.predict_proba([embedding])[0]
        best_idx = int(np.argmax(probabilities))
        best_prob = float(probabilities[best_idx])
        best_label = str(self.label_encoder.inverse_transform([best_idx])[0])

        if nearest_distance > threshold:
            return Prediction("desconocido", 0.0, nearest_distance, method="distancia")

        if best_label != nearest_label and best_prob >= self.probability_threshold:
            # El SVM y el vecino mas cercano no coinciden: aceptar con cautela.
            cautious_confidence = min(calibrated_confidence, best_prob) * 0.75
            return Prediction(best_label, cautious_confidence, nearest_distance, method="svm")

        confidence = max(calibrated_confidence, min(best_prob, 0.95) if best_label == nearest_label else 0.0)
        return Prediction(nearest_label, confidence, nearest_distance, method="calibrada")

    def _ensure_runtime_defaults(self) -> None:
        # Compatibilidad con modelos joblib entrenados antes de agregar calibracion.
        if not hasattr(self, "reference_embeddings") or self.reference_embeddings is None:
            self.reference_embeddings = np.vstack(list(self.centroids.values())).astype(np.float32)
            self.reference_labels = list(self.centroids.keys())
        if not hasattr(self, "class_thresholds"):
            self.class_thresholds = {}
        if not hasattr(self, "class_core_distances"):
            self.class_core_distances = {}
        if not hasattr(self, "probability_threshold"):
            self.probability_threshold = 0.55

    def _nearest_centroid(self, embedding: np.ndarray) -> Tuple[str, float]:
        distances = {
            label: float(np.linalg.norm(embedding - centroid))
            for label, centroid in self.centroids.items()
        }
        label = min(distances, key=distances.get)
        return label, distances[label]

    def _nearest_reference(self, embedding: np.ndarray) -> Tuple[str, float]:
        if self.reference_embeddings is None or len(self.reference_labels) == 0:
            return self._nearest_centroid(embedding)

        distances = np.linalg.norm(self.reference_embeddings - embedding, axis=1)
        idx = int(np.argmin(distances))
        return self.reference_labels[idx], float(distances[idx])

    def _calibrated_confidence(self, label: str, distance: float) -> float:
        threshold = self.class_thresholds.get(label, self.distance_threshold)
        core = self.class_core_distances.get(label, threshold * 0.55)
        if distance <= core:
            return 0.99
        if distance >= threshold:
            return 0.0
        confidence = 1.0 - ((distance - core) / max(threshold - core, 1e-6))
        return float(np.clip(confidence, 0.0, 0.99))


def train_classifier(embeddings: np.ndarray, labels: List[str]) -> FaceSVMClassifier:
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(labels)
    unique_labels = sorted(set(labels))

    centroids = {
        label: embeddings[np.array(labels) == label].mean(axis=0)
        for label in unique_labels
    }
    centroids = {
        label: centroid / max(np.linalg.norm(centroid), 1e-6)
        for label, centroid in centroids.items()
    }

    class_core_distances, class_thresholds = _calibrate_class_distances(embeddings, labels)
    threshold = float(max(class_thresholds.values())) if class_thresholds else 0.95
    pipeline = None
    if len(unique_labels) >= 2:
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)),
            ]
        )
        pipeline.fit(embeddings, encoded)

    return FaceSVMClassifier(
        pipeline=pipeline,
        label_encoder=label_encoder,
        centroids=centroids,
        distance_threshold=threshold,
        reference_embeddings=embeddings.astype(np.float32),
        reference_labels=list(labels),
        class_thresholds=class_thresholds,
        class_core_distances=class_core_distances,
    )


def _calibrate_class_distances(
    embeddings: np.ndarray,
    labels: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    labels_array = np.array(labels)
    core_distances: Dict[str, float] = {}
    thresholds: Dict[str, float] = {}

    for label in sorted(set(labels)):
        class_embeddings = embeddings[labels_array == label]
        if len(class_embeddings) <= 1:
            core_distances[label] = 0.35
            thresholds[label] = 0.80
            continue

        leave_one_out_distances = []
        for idx, embedding in enumerate(class_embeddings):
            others = np.delete(class_embeddings, idx, axis=0)
            distances = np.linalg.norm(others - embedding, axis=1)
            leave_one_out_distances.append(float(np.min(distances)))

        median_distance = float(np.median(leave_one_out_distances))
        p95_distance = float(np.percentile(leave_one_out_distances, 95))
        core_distances[label] = float(np.clip(median_distance, 0.20, 0.75))
        thresholds[label] = float(np.clip(p95_distance + 0.18, 0.55, 1.05))

    return core_distances, thresholds


def save_classifier(classifier: FaceSVMClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, path)


def load_classifier(path: Path) -> FaceSVMClassifier:
    return joblib.load(path)

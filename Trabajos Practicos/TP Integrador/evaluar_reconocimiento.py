from pathlib import Path
import argparse
import json
import sys
from typing import Dict, List, Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from tp_integrador.backend.celebrity import CelebrityIndex
from tp_integrador.backend.clasificador import Prediction, train_classifier
from tp_integrador.backend.data import EMBEDDINGS_DIR


OUTPUT_PATH = ROOT_DIR / "datos" / "evaluacion_reconocimiento.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua reconocimiento facial con split train/test y negativos.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraccion de embeddings propios reservada para test.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para hacer el split reproducible.")
    parser.add_argument("--unknown-limit", type=int, default=300, help="Maximo de embeddings de famosos usados como desconocidos.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Ruta del JSON de salida.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_embedding_records()
    if len(records) < 2:
        raise SystemExit("No hay suficientes embeddings propios para evaluar.")

    train_records, test_records = split_records(records, args.test_size, args.seed)
    if not test_records:
        raise SystemExit("No se pudo reservar test. Captura al menos 2 embeddings por persona.")

    train_embeddings = np.vstack([record[2] for record in train_records]).astype(np.float32)
    train_labels = [record[1] for record in train_records]
    classifier = train_classifier(train_embeddings, train_labels)

    known_results = evaluate_known(classifier, test_records)
    unknown_records = load_unknown_records(args.unknown_limit, args.seed)
    unknown_results = evaluate_unknown(classifier, unknown_records)

    report = build_report(train_records, test_records, known_results, unknown_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(report, args.output)


def load_embedding_records() -> List[Tuple[str, str, np.ndarray]]:
    records = []
    for path in sorted(EMBEDDINGS_DIR.glob("*/*.npz")):
        data = np.load(path, allow_pickle=False)
        embedding = normalize(data["embedding"].astype(np.float32))
        records.append((str(path), str(data["name"]), embedding))
    return records


def split_records(
    records: List[Tuple[str, str, np.ndarray]],
    test_size: float,
    seed: int,
) -> Tuple[List[Tuple[str, str, np.ndarray]], List[Tuple[str, str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    by_label: Dict[str, List[Tuple[str, str, np.ndarray]]] = {}
    for record in records:
        by_label.setdefault(record[1], []).append(record)

    train_records = []
    test_records = []
    for label_records in by_label.values():
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_records.extend(shuffled)
            continue
        test_count = int(round(len(shuffled) * test_size))
        test_count = min(max(1, test_count), len(shuffled) - 1)
        test_records.extend(shuffled[:test_count])
        train_records.extend(shuffled[test_count:])
    return train_records, test_records


def evaluate_known(classifier, records: List[Tuple[str, str, np.ndarray]]) -> List[dict]:
    results = []
    for path, expected_label, embedding in records:
        prediction = classifier.predict(embedding)
        results.append(prediction_to_result(path, expected_label, prediction))
    return results


def evaluate_unknown(classifier, records: List[Tuple[str, str, np.ndarray]]) -> List[dict]:
    results = []
    for path, _, embedding in records:
        prediction = classifier.predict(embedding)
        results.append(prediction_to_result(path, "desconocido", prediction))
    return results


def load_unknown_records(limit: int, seed: int) -> List[Tuple[str, str, np.ndarray]]:
    if limit <= 0 or not CelebrityIndex.exists():
        return []
    index = CelebrityIndex.load()
    embeddings = index.sample_embeddings if index.sample_embeddings is not None else index.embeddings
    image_paths = index.sample_image_paths if index.sample_image_paths else index.image_paths
    if embeddings is None or embeddings.size == 0:
        return []

    rng = np.random.default_rng(seed)
    count = min(limit, len(embeddings))
    selected = rng.choice(len(embeddings), size=count, replace=False)
    records = []
    for idx in selected:
        path = str(image_paths[int(idx)]) if int(idx) < len(image_paths) else f"celebrity:{idx}"
        records.append((path, "desconocido", normalize(embeddings[int(idx)].astype(np.float32))))
    return records


def prediction_to_result(path: str, expected_label: str, prediction: Prediction) -> dict:
    return {
        "path": path,
        "expected": expected_label,
        "predicted": prediction.label,
        "confidence": prediction.confidence,
        "distance": prediction.distance,
        "method": prediction.method,
        "ok": prediction.label == expected_label,
    }


def build_report(train_records, test_records, known_results: List[dict], unknown_results: List[dict]) -> dict:
    known_ok = sum(1 for result in known_results if result["ok"])
    known_rejected = sum(1 for result in known_results if result["predicted"] == "desconocido")
    unknown_ok = sum(1 for result in unknown_results if result["ok"])
    false_accepts = [result for result in unknown_results if result["predicted"] != "desconocido"]
    confusion: Dict[str, Dict[str, int]] = {}
    for result in known_results + unknown_results:
        confusion.setdefault(result["expected"], {})
        confusion[result["expected"]][result["predicted"]] = confusion[result["expected"]].get(result["predicted"], 0) + 1

    return {
        "samples": {
            "train": len(train_records),
            "known_test": len(known_results),
            "unknown_test": len(unknown_results),
        },
        "metrics": {
            "known_accuracy": safe_ratio(known_ok, len(known_results)),
            "known_false_unknown_rate": safe_ratio(known_rejected, len(known_results)),
            "unknown_rejection_rate": safe_ratio(unknown_ok, len(unknown_results)),
            "unknown_false_accept_rate": safe_ratio(len(false_accepts), len(unknown_results)),
        },
        "confusion": confusion,
        "known_results": known_results,
        "unknown_false_accepts": false_accepts[:50],
    }


def print_summary(report: dict, output_path: Path) -> None:
    metrics = report["metrics"]
    samples = report["samples"]
    print(f"Train: {samples['train']} | Test conocidos: {samples['known_test']} | Desconocidos: {samples['unknown_test']}")
    print(f"Accuracy conocidos: {metrics['known_accuracy']:.3f}")
    print(f"Falsos desconocidos: {metrics['known_false_unknown_rate']:.3f}")
    print(f"Rechazo desconocidos: {metrics['unknown_rejection_rate']:.3f}")
    print(f"Falsos aceptados: {metrics['unknown_false_accept_rate']:.3f}")
    print(f"Reporte guardado en: {output_path}")


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def normalize(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype(np.float32)


if __name__ == "__main__":
    main()

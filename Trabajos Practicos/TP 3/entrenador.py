import argparse
import json

import matplotlib.pyplot as plt

from models.tp3_models import ModeloBase, ModeloOptimizado
from utils.tp3_config import MODEL_PATHS, TRAIN_OUTPUT_DIR
from utils.tp3_data import build_loaders
from utils.tp3_history import plot_history, save_history
from utils.tp3_evaluation import evaluate_dataset, show_random_predictions
from utils.tp3_training import evaluate_saved_model, train_model


def build_model(mode):
    if mode == "optimized":
        return ModeloOptimizado()
    return ModeloBase()


def use_data_augmentation(mode):
    return mode in {"augmentation", "optimized"}


def build_model_name(mode):
    if mode == "optimized":
        return "modelo_optimizado"
    return f"modelo_{mode}"


def print_step(step_number, text):
    print(f"\nPaso {step_number}: {text}")


def describe_improvements(mode):
    print("Tecnicas para mejorar metricas:")
    if use_data_augmentation(mode):
        print("- data augmentation (flip, rotacion, zoom, brillo)")
    if mode == "optimized":
        print("- modelo mas profundo (mas capas conv)")
    if mode == "base":
        print("- modelo base para comparar")


def plot_comparison(results, output_path=TRAIN_OUTPUT_DIR / "comparacion_resultados.png"):
    names = [item["nombre"] for item in results]
    accuracies = [item["test_acc"] for item in results]
    losses = [item["test_loss"] for item in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(names, accuracies, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    axes[0].set_title("Accuracy en test")
    axes[0].set_ylim(0, 1)

    axes[1].bar(names, losses, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    axes[1].set_title("Loss en test")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_experiment(mode, run_demo=True):
    print_step(1, "Cargar dataset (train/val/test)")
    train_ds, val_ds, test_ds = build_loaders(use_augmentation=use_data_augmentation(mode))

    print_step(2, "Elegir arquitectura y tecnicas")
    model = build_model(mode)
    model_name = build_model_name(mode)
    describe_improvements(mode)
    TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_step(3, "Entrenar y registrar metricas (accuracy y loss)")
    history = train_model(model, train_ds, val_ds, model_name)

    print_step(4, "Evaluar con el conjunto de test")
    test_loss, test_acc = evaluate_saved_model(model, test_ds)
    if run_demo:
        evaluate_dataset(model)
        print("Matriz de confusion guardada en salida evaluacion/")

    if run_demo:
        print_step(5, "Ejecutar el modelo con imagenes de prueba")
        show_random_predictions(model, quantity=6)
        print("Predicciones guardadas en salida evaluacion/")

    result = {
        "nombre": model_name,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "epochs": len(history["train_loss"]),
    }

    save_history(history, TRAIN_OUTPUT_DIR / f"{model_name}_history.json")
    plot_history(history, model_name, TRAIN_OUTPUT_DIR / f"{model_name}_history.png")

    print(f"\nResultado {model_name}: acc={test_acc:.4f} loss={test_loss:.4f}")
    print(f"Modelo guardado en {MODEL_PATHS[mode].name}")
    return result


def train_all_models():
    results = []
    for mode in ["base", "augmentation", "optimized"]:
        results.append(run_experiment(mode, run_demo=False))

    TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_OUTPUT_DIR / "resultados_entrenamiento.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    plot_comparison(results)
    print("\nResultados guardados en salida entrenamiento/")


def main():
    parser = argparse.ArgumentParser(description="Entrenador TP 3")
    parser.add_argument(
        "--modo",
        choices=["base", "augmentation", "optimized", "todos"],
        default="todos",
        help="Que modelo(s) entrenar: base, augmentation, optimized, o todos los 3",
    )
    args = parser.parse_args()

    if args.modo == "todos":
        train_all_models()
        return

    run_experiment(args.modo)


if __name__ == "__main__":
    main()

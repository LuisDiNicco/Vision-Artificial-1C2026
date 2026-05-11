import argparse
import json

import matplotlib.pyplot as plt

from utils.tp3_config import DEVICE, MODEL_PATHS, TRAIN_OUTPUT_DIR
from utils.tp3_data import build_loaders
from models.tp3_models import ModeloBase, ModeloOptimizado
from utils.tp3_training import evaluate_saved_model, plot_history, save_history, train_model


def build_model(mode):
    if mode == "optimized":
        return ModeloOptimizado().to(DEVICE), True
    return ModeloBase().to(DEVICE), mode == "augmentation"


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


def run_experiment(mode):
    use_augmentation = mode in {"augmentation", "optimized"}
    train_loader, test_loader = build_loaders(use_augmentation=use_augmentation)
    model, use_scheduler = build_model(mode)
    model_name = f"modelo_{mode}"
    TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history = train_model(model, train_loader, test_loader, model_name, use_scheduler=use_scheduler)
    test_loss, test_acc = evaluate_saved_model(model, test_loader)

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


def main():
    parser = argparse.ArgumentParser(description="Entrenador TP 3")
    parser.add_argument(
        "--modo",
        choices=["base", "augmentation", "optimized", "todos"],
        default="todos",
        help="Qué modelo(s) entrenar: base, augmentation, optimized, o todos los 3",
    )
    args = parser.parse_args()

    if args.modo == "todos":
        results = [run_experiment(mode) for mode in ["base", "augmentation", "optimized"]]
        TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRAIN_OUTPUT_DIR / "resultados_entrenamiento.json", "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
        plot_comparison(results)
        print("\nResultados guardados en salida entrenamiento/")
        return

    run_experiment(args.modo)


if __name__ == "__main__":
    main()
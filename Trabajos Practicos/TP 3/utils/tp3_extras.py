import json

import matplotlib.pyplot as plt


def print_step(step_number, text):
    # Mostrar el flujo como pasos, tipo notebook/colab.
    print(f"\nPaso {step_number}: {text}")


def describe_improvements(mode, use_augmentation):
    # Explicamos las tecnicas usadas segun el modo.
    print("Tecnicas para mejorar metricas:")
    if use_augmentation:
        print("- data augmentation (flip, rotacion, zoom, brillo)")
    if mode == "optimized":
        print("- modelo mas profundo (mas capas conv)")
    if mode == "base":
        print("- modelo base para comparar")


def print_progress_bar(prefix, current, total):
    # Barra de progreso en una sola linea.
    if total <= 0:
        return
    percent = int((current / total) * 100)
    bar_len = 20
    filled = int(bar_len * current / total)
    bar = "#" * filled + "." * (bar_len - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({percent}%)", end="")


def clear_progress_bar():
    # Limpia la linea para que no queden barras anteriores.
    print("\r" + " " * 80 + "\r", end="")


def save_history(history, output_path):
    # Guarda el historial para poder explicar el proceso despues.
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def save_results(results, output_path):
    # Guarda el resumen final de resultados en JSON.
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def plot_history(history, title, output_path):
    # Grafica accuracy y loss por epoca para ver si el modelo aprende.
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_acc"], label="train")
    axes[0].plot(epochs, history["val_acc"], label="val")
    axes[0].set_title(f"Accuracy - {title}")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], label="train")
    axes[1].plot(epochs, history["val_loss"], label="val")
    axes[1].set_title(f"Loss - {title}")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results, output_path):
    # Grafico simple para comparar resultados finales.
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


def save_model_reports(history, model_name, output_dir):
    # Guarda JSON y grafico para un modelo.
    save_history(history, output_dir / f"{model_name}_history.json")
    plot_history(history, model_name, output_dir / f"{model_name}_history.png")


def save_experiment_reports(results, output_dir):
    # Guarda el resumen general y el grafico comparativo.
    save_results(results, output_dir / "resultados_entrenamiento.json")
    plot_comparison(results, output_dir / "comparacion_resultados.png")

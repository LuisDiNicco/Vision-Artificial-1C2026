import json

import matplotlib.pyplot as plt

# Imprime un paso del flujo numerado (para presentacion)
def print_step(step_number, text):
    # Formato: "\nPaso N: descripcion"
    print(f"\nPaso {step_number}: {text}")


# Describe que tecnicas se usan segun el modo (base, augmentation, optimized)
def describe_improvements(mode, use_augmentation):
    # Explica cuales mejoras se aplican
    print("Tecnicas para mejorar metricas:")
    if use_augmentation:
        print("- data augmentation (flip, rotacion, zoom, brillo)")
    if mode == "optimized":
        print("- modelo mas profundo (mas capas conv)")
    if mode == "base":
        print("- modelo base para comparar")


# Imprime una barra de progreso en linea unica (no ocupa multiples lineas)
def print_progress_bar(prefix, current, total):
    # Muestra formato: "Entrenando [####....] 50/100 (50%)"
    if total <= 0:
        return
    percent = int((current / total) * 100)
    bar_len = 20
    filled = int(bar_len * current / total)
    bar = "#" * filled + "." * (bar_len - filled)
    # \r retrocede al inicio de la linea para sobreescribir
    print(f"\r{prefix} [{bar}] {current}/{total} ({percent}%)", end="")


# Limpia la linea de barra de progreso
def clear_progress_bar():
    # Imprime espacios en blanco para dejar linea limpia
    print("\r" + " " * 80 + "\r", end="")


# Guarda el historial de entrenamiento en JSON
def save_history(history, output_path):
    # Guarda metricas (loss y accuracy) para poder analizar despues
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


# Guarda los resultados finales de todos los modelos
def save_results(results, output_path):
    # Guarda resumen JSON con accuracy y loss final de cada modelo
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


# Grafica los graficos de accuracy y loss de un modelo
def plot_history(history, title, output_path):
    # Grafica 2 subplots: accuracy y loss vs epocas
    # Util para visualizar si el modelo esta aprendiendo
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy: muestra train vs val
    axes[0].plot(epochs, history["train_acc"], label="train")
    axes[0].plot(epochs, history["val_acc"], label="val")
    axes[0].set_title(f"Accuracy - {title}")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()

    # Loss: muestra train vs val (deberia disminuir)
    axes[1].plot(epochs, history["train_loss"], label="train")
    axes[1].plot(epochs, history["val_loss"], label="val")
    axes[1].set_title(f"Loss - {title}")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Grafica comparacion de resultados finales entre los 3 modelos
def plot_comparison(results, output_path):
    # Extrae metricas finales de cada modelo para comparar
    names = [item["nombre"] for item in results]
    accuracies = [item["test_acc"] for item in results]
    losses = [item["test_loss"] for item in results]

    # Crea 2 graficos: barras de accuracy y barras de loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Colores diferentes para cada modelo
    axes[0].bar(names, accuracies, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    axes[0].set_title("Accuracy en test")
    axes[0].set_ylim(0, 1)

    axes[1].bar(names, losses, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    axes[1].set_title("Loss en test")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Guarda JSON y grafico del historial para un modelo
def save_model_reports(history, model_name, output_dir):
    # Guarda en JSON: accuracy y loss por cada epoca
    save_history(history, output_dir / f"{model_name}_history.json")
    # Guarda en PNG: dos graficos (accuracy y loss)
    plot_history(history, model_name, output_dir / f"{model_name}_history.png")


# Guarda reportes generales de todos los experimentos
def save_experiment_reports(results, output_dir):
    # Guarda JSON con resumen final (accuracy y loss de cada modelo)
    save_results(results, output_dir / "resultados_entrenamiento.json")
    # Guarda PNG con grafico comparativo entre modelos
    plot_comparison(results, output_dir / "comparacion_resultados.png")

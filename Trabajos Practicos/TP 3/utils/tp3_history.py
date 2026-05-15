import json

import matplotlib.pyplot as plt


def save_history(history, output_path):
    # Guarda el historial para poder explicar el proceso despues.
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


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

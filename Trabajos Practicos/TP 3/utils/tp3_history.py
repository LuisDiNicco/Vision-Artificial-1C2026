import json

import matplotlib.pyplot as plt

# Guarda el historial de entrenamiento en JSON para poder analizar despues
def save_history(history, output_path):
    # Recibe diccionario con train_loss, train_acc, val_loss, val_acc
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


# Grafica accuracy y loss de train/val por epoca para visualizar el aprendizaje
def plot_history(history, title, output_path):
    # Rango de epocas para el eje X
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Grafico 1: Accuracy por epoca
    # Azul = train, Naranja = val
    # Si train >> val = overfitting. Si estan juntas = buen balance
    axes[0].plot(epochs, history["train_acc"], label="train")
    axes[0].plot(epochs, history["val_acc"], label="val")
    axes[0].set_title(f"Accuracy - {title}")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()

    # Grafico 2: Loss por epoca
    # Deberia disminuir con las epocas (aprende)
    axes[1].plot(epochs, history["train_loss"], label="train")
    axes[1].plot(epochs, history["val_loss"], label="val")
    axes[1].set_title(f"Loss - {title}")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms

from utils.tp3_config import CLASS_NAMES_ES, DEVICE, EVAL_OUTPUT_DIR, IMG_SIZE, MODEL_PATHS, TEST_PATH, USE_PIN_MEMORY, get_num_workers
from utils.runtime.device_utils import safe_torch_load
from models.tp3_models import ModeloBase, ModeloOptimizado


def build_model_from_path(model_path):
    # Elegimos la arquitectura segun el nombre del archivo.
    if "optimized" in model_path or "optimizado" in model_path:
        return ModeloOptimizado().to(DEVICE)
    return ModeloBase().to(DEVICE)


def load_model(model_path):
    # Cargamos el modelo guardado en disco.
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = build_model_from_path(model_path)

    state_dict = safe_torch_load(model_path, DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_test_transform():
    # Transformaciones basicas para evaluar.
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_image(model, image_path):
    # Predecimos una sola imagen.
    image = Image.open(image_path).convert("RGB")
    transformed = build_test_transform()(image).unsqueeze(0).to(DEVICE)

    # Evaluacion simple, sin gradientes.
    with torch.no_grad():
        outputs = model(transformed)

    probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    return probabilities, predicted_index, confidence, image


def evaluate_dataset(model):
    # Evaluamos todo el conjunto de prueba.
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = datasets.ImageFolder(TEST_PATH, transform=build_test_transform())
    num_workers = get_num_workers()
    loader_kwargs = {
        "batch_size": 32,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": USE_PIN_MEMORY,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    all_predictions = []
    all_labels = []

    # Evaluacion simple, sin gradientes.
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            all_predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Calculamos el porcentaje de aciertos.
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"\nAccuracy total: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(classification_report(all_labels, all_predictions, target_names=CLASS_NAMES_ES))

    # Matriz de confusion para ver en que clases se equivoca.
    matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES_ES, yticklabels=CLASS_NAMES_ES)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusion")
    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_DIR / "matriz_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()

def show_random_predictions(model, quantity=6):
    # Mostramos algunas predicciones al azar.
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = datasets.ImageFolder(TEST_PATH, transform=build_test_transform())
    indices = random.sample(range(len(dataset.samples)), k=min(quantity, len(dataset.samples)))

    rows = (len(indices) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for axis, index in zip(axes, indices):
        image_path, label_index = dataset.samples[index]
        probabilities, predicted_index, confidence, image = predict_image(model, image_path)
        correct = predicted_index == label_index
        color = "green" if correct else "red"
        axis.imshow(image)
        axis.set_title(
            f"Real: {CLASS_NAMES_ES[label_index]}\nPred: {CLASS_NAMES_ES[predicted_index]}\nConf: {confidence:.2%}",
            color=color,
            fontweight="bold",
        )
        axis.axis("off")

    for axis in axes[len(indices):]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_DIR / "predicciones_aleatorias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    # Elegimos el modelo a evaluar.
    parser = argparse.ArgumentParser(description="Evaluador TP 3")
    parser.add_argument("--modelo", choices=["base", "augmentation", "optimized"], default="optimized")
    args = parser.parse_args()

    model_path = str(MODEL_PATHS[args.modelo])
    model = load_model(model_path)

    print(f"Modelo cargado: {model_path}")
    print("1. Evaluar conjunto de prueba")
    print("2. Ver predicciones aleatorias")
    print("3. Predecir una imagen")

    # Menu simple para elegir la accion.
    option = input("Opcion: ").strip()
    if option == "1":
        evaluate_dataset(model)
    elif option == "2":
        amount = input("Cantidad de imagenes: ").strip()
        quantity = int(amount) if amount.isdigit() else 6
        show_random_predictions(model, quantity)
    elif option == "3":
        image_path = input("Ruta de la imagen: ").strip()
        if os.path.exists(image_path):
            probabilities, predicted_index, confidence, image = predict_image(model, image_path)
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.barh(CLASS_NAMES_ES, probabilities)
            plt.title(f"Prediccion: {CLASS_NAMES_ES[predicted_index]} ({confidence:.2%})")
            plt.tight_layout()
            plt.show()
        else:
            print("Archivo no encontrado")


if __name__ == "__main__":
    main()
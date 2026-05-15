import argparse
import os

import matplotlib.pyplot as plt

from utils.tp3_config import CLASS_NAMES_ES, MODEL_PATHS
from utils.tp3_evaluation import evaluate_dataset, load_model, predict_image, show_random_predictions


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
import argparse
import logging
import os

# Silencia logs ruidosos de TensorFlow antes de importar utilidades
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import matplotlib.pyplot as plt

from utils.tp3_config import BASE_DIR, CLASS_NAMES_ES, MODEL_PATHS
from utils.tp3_evaluation import evaluate_dataset, load_model, predict_image, show_random_predictions


def _collect_images(folder_path):
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for name in files:
            _, ext = os.path.splitext(name)
            if ext.lower() in valid_ext:
                image_paths.append(os.path.join(root, name))
    return sorted(image_paths)


def _show_prediction_browser(model, image_paths):
    if not image_paths:
        print("No se encontraron imagenes para mostrar.")
        return

    index = 0
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    def render(idx):
        probabilities, predicted_index, confidence, image = predict_image(model, image_paths[idx])
        axes[0].clear()
        axes[0].imshow(image)
        axes[0].set_title(f"Imagen {idx + 1}/{len(image_paths)}")
        axes[0].axis("off")

        axes[1].clear()
        axes[1].barh(CLASS_NAMES_ES, probabilities)
        axes[1].set_xlim(0, 1)
        axes[1].set_title(
            f"Prediccion: {CLASS_NAMES_ES[predicted_index]} ({confidence:.2%})"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal index
        if event.key in {"right", "d"}:
            index = (index + 1) % len(image_paths)
            render(index)
        elif event.key in {"left", "a"}:
            index = (index - 1) % len(image_paths)
            render(index)
        elif event.key in {"escape", "q"}:
            plt.close(fig)

    render(index)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Usa flechas izquierda/derecha (o A/D) para navegar. Q o ESC para salir.")
    plt.show()


# Punto de entrada del programa de evaluacion
def main():
    # Parsea el modelo a usar (base, augmentation u optimized)
    parser = argparse.ArgumentParser(description="Evaluador TP 3")
    parser.add_argument("--modelo", choices=["base", "augmentation", "optimized"], default="optimized")
    args = parser.parse_args()

    # Carga el modelo entrenado
    model_path = str(MODEL_PATHS[args.modelo])
    model = load_model(model_path)

    print(f"Modelo cargado: {model_path}")
    # Menu interactivo con 3 opciones
    print("1. Evaluar conjunto de prueba")
    print("2. Ver predicciones aleatorias")
    print("3. Predecir una imagen")

    option = input("Opcion: ").strip()
    if option == "1":
        # Opcion 1: Evalua todo el conjunto de test (accuracy, loss, matriz confusion)
        evaluate_dataset(model)
    elif option == "2":
        # Opcion 2: Muestra N predicciones aleatorias
        amount = input("Cantidad de imagenes: ").strip()
        quantity = int(amount) if amount.isdigit() else 6
        show_random_predictions(model, quantity)
    elif option == "3":
        # Opcion 3: Predice imagenes con navegacion por carpeta
        default_dir = os.path.join(BASE_DIR, "Imaganes de Paisajes", "seg_pred", "seg_pred")
        prompt = "Ruta de la imagen o carpeta (Enter = seg_pred): "
        user_path = input(prompt).strip()
        target_path = user_path if user_path else default_dir

        if os.path.isdir(target_path):
            image_paths = _collect_images(target_path)
            _show_prediction_browser(model, image_paths)
        elif os.path.isfile(target_path):
            _show_prediction_browser(model, [target_path])
        else:
            print("Archivo o carpeta no encontrada")


if __name__ == "__main__":
    main()

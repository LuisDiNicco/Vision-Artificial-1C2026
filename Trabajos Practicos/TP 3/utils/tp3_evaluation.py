import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from .tp3_config import (
    CLASS_NAMES_ES,
    EVAL_OUTPUT_DIR,
    IMG_SIZE,
    TEST_PATH,
)
from .tp3_data import IMAGENET_MEAN, IMAGENET_STD


# Carga un modelo entrenado desde disco
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    return keras.models.load_model(model_path)


# Preprocesa una imagen para que el modelo pueda predecir
def _preprocess_image(image):
    # Redimensiona a IMG_SIZE x IMG_SIZE
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # Normaliza: escala a [0,1] y aplica normalizacion ImageNet
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant(IMAGENET_MEAN)
    std = tf.constant(IMAGENET_STD)
    image = (image - mean) / std
    return image


# Predice la clase de una imagen individual
def predict_image(model, image_path):
    # Carga la imagen
    image = Image.open(image_path).convert("RGB")
    image_array = tf.constant(np.array(image), dtype=tf.float32)
    # Preprocesa la imagen
    processed = _preprocess_image(image_array)
    # Agrega batch dimension (1, IMG_SIZE, IMG_SIZE, 3)
    processed = tf.expand_dims(processed, 0)

    # Realiza la prediccion (logits sin softmax)
    outputs = model(processed, training=False)
    # Aplica softmax para obtener probabilidades
    probabilities = tf.nn.softmax(outputs, axis=1)[0].numpy()
    # Obtiene la clase predicha
    predicted_index = int(np.argmax(probabilities))
    # Obtiene la confianza (probabilidad de la clase predicha)
    confidence = float(probabilities[predicted_index])
    return probabilities, predicted_index, confidence, image


# Evalua el modelo en todo el dataset de test
# Calcula accuracy, classification report y genera matriz de confusion
def evaluate_dataset(model):
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Carga el dataset de test
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEST_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        label_mode='int',
        shuffle=False,
    )

    # Recolecta todas las predicciones y etiquetas reales
    all_predictions = []
    all_labels = []

    # Itera sobre todos los batches
    for images, labels in test_ds:
        # Normaliza imagenes
        images = tf.cast(images, tf.float32) / 255.0
        mean = tf.constant(IMAGENET_MEAN)
        std = tf.constant(IMAGENET_STD)
        images = (images - mean) / std

        # Realiza predicciones
        outputs = model(images, training=False)
        # Obtiene clase predicha (argmax de logits)
        preds = tf.argmax(outputs, axis=1).numpy()
        all_predictions.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    # Calcula accuracy global
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"\nAccuracy total: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    # Muestra reporte de clasificacion (precision, recall, f1 por clase)
    print(classification_report(all_labels, all_predictions, target_names=CLASS_NAMES_ES))

    # Genera y guarda matriz de confusion
    matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES_ES, yticklabels=CLASS_NAMES_ES)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusion")
    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_DIR / "matriz_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()


# Muestra N predicciones aleatorias con sus probabilidades
# Resalta en verde aciertos y en rojo errores
def show_random_predictions(model, quantity=6):
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Carga el dataset de test
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEST_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        label_mode='int',
        shuffle=False,
    )
    # Obtiene las rutas de todas las imagenes
    file_paths = test_ds.file_paths
    # Selecciona N indices aleatorios
    indices = random.sample(range(len(file_paths)), k=min(quantity, len(file_paths)))

    # Obtiene nombres de clases del dataset
    class_names = test_ds.class_names

    # Calcula layout de subplots (3 columnas)
    rows = (len(indices) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)

    # Itera sobre los indices seleccionados
    for axis, index in zip(axes, indices):
        # Obtiene la ruta y etiqueta real de la imagen
        image_path = file_paths[index]
        parent_folder = os.path.basename(os.path.dirname(image_path))
        label_index = class_names.index(parent_folder)

        # Realiza prediccion
        probabilities, predicted_index, confidence, image = predict_image(model, image_path)
        # Verifica si la prediccion es correcta
        correct = predicted_index == label_index
        # Usa color verde para aciertos, rojo para errores
        color = "green" if correct else "red"
        # Muestra la imagen con titulo que indica etiqueta real, predicha y confianza
        axis.imshow(image)
        axis.set_title(
            f"Real: {CLASS_NAMES_ES[label_index]}\nPred: {CLASS_NAMES_ES[predicted_index]}\nConf: {confidence:.2%}",
            color=color,
            fontweight="bold",
        )
        axis.axis("off")

    # Oculta subplots vacios
    for axis in axes[len(indices):]:
        axis.axis("off")

    # Guarda el grafico
    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_DIR / "predicciones_aleatorias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

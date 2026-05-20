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


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    return keras.models.load_model(model_path)


def _preprocess_image(image):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant(IMAGENET_MEAN)
    std = tf.constant(IMAGENET_STD)
    image = (image - mean) / std
    return image


def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_array = tf.constant(np.array(image), dtype=tf.float32)
    processed = _preprocess_image(image_array)
    processed = tf.expand_dims(processed, 0)

    outputs = model(processed, training=False)
    probabilities = tf.nn.softmax(outputs, axis=1)[0].numpy()
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    return probabilities, predicted_index, confidence, image


def evaluate_dataset(model):
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEST_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        label_mode='int',
        shuffle=False,
    )

    all_predictions = []
    all_labels = []

    for images, labels in test_ds:
        images = tf.cast(images, tf.float32) / 255.0
        mean = tf.constant(IMAGENET_MEAN)
        std = tf.constant(IMAGENET_STD)
        images = (images - mean) / std

        outputs = model(images, training=False)
        preds = tf.argmax(outputs, axis=1).numpy()
        all_predictions.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"\nAccuracy total: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(classification_report(all_labels, all_predictions, target_names=CLASS_NAMES_ES))

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
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEST_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        label_mode='int',
        shuffle=False,
    )
    file_paths = test_ds.file_paths
    indices = random.sample(range(len(file_paths)), k=min(quantity, len(file_paths)))

    class_names = test_ds.class_names

    rows = (len(indices) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for axis, index in zip(axes, indices):
        image_path = file_paths[index]
        parent_folder = os.path.basename(os.path.dirname(image_path))
        label_index = class_names.index(parent_folder)

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

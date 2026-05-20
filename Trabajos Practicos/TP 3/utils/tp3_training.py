import time
from pathlib import Path

from tensorflow import keras

from .tp3_config import EPOCHS, LEARNING_RATE, BASE_DIR


# Entrena un modelo y guarda el mejor segun loss de validacion
def train_model(model, train_ds, val_ds, model_name, epochs=EPOCHS):
    # Crea directorio para guardar modelos
    models_dir = BASE_DIR / "modelos_guardados"
    models_dir.mkdir(parents=True, exist_ok=True)
    final_path = str(models_dir / f"{model_name}.keras")

    # Compila el modelo con optimizador SGD
    model.compile(
        # SGD con momentum para convergencia mas estable
        optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9),
        # Loss para clasificacion multiclase
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    # Callbacks durante el entrenamiento
    callbacks = [
        # Early stopping: para si el loss de validacion no mejora (paciencia=4 epocas)
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        # Model checkpoint: guarda el modelo cada vez que mejora val_loss
        keras.callbacks.ModelCheckpoint(
            final_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0,
        ),
    ]

    print(f"\nEntrenando {model_name}")
    print("Optimizador: SGD")

    # Mide el tiempo de entrenamiento
    start_time = time.time()

    # Entrena el modelo
    history_obj = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Calcula tiempo de entrenamiento
    elapsed = time.time() - start_time
    print(f"Terminado en {elapsed / 60:.2f} minutos")

    # Extrae el historial de metricas
    history = {
        "train_loss": history_obj.history['loss'],
        "train_acc": history_obj.history['accuracy'],
        "val_loss": history_obj.history['val_loss'],
        "val_acc": history_obj.history['val_accuracy'],
    }
    return history


# Evalua el modelo en el dataset de test
def evaluate_saved_model(model, test_ds):
    # Retorna loss y accuracy en el conjunto de test
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    return loss, accuracy

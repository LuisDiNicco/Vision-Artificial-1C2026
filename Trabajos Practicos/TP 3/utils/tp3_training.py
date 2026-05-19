import time
from pathlib import Path

from tensorflow import keras

from .tp3_config import EPOCHS, LEARNING_RATE


def train_model(model, train_ds, val_ds, model_name, epochs=EPOCHS):
    models_dir = Path("modelos_guardados")
    models_dir.mkdir(parents=True, exist_ok=True)
    final_path = str(models_dir / f"{model_name}.keras")

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            final_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0,
        ),
    ]

    print(f"\nEntrenando {model_name}")
    print("Optimizador: SGD")

    start_time = time.time()

    history_obj = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    elapsed = time.time() - start_time
    print(f"Terminado en {elapsed / 60:.2f} minutos")

    history = {
        "train_loss": history_obj.history['loss'],
        "train_acc": history_obj.history['accuracy'],
        "val_loss": history_obj.history['val_loss'],
        "val_acc": history_obj.history['val_accuracy'],
    }
    return history


def evaluate_saved_model(model, test_ds):
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    return loss, accuracy

import tensorflow as tf
from tensorflow import keras
from keras import layers

from .tp3_config import BATCH_SIZE, IMG_SIZE, TEST_PATH, TRAIN_PATH


# Construye pipeline de data augmentation
# Aplica transformaciones aleatorias para aumentar variedad en datos de entrenamiento
def _build_augmentation():
    return keras.Sequential([
        # Voltea horizontalmente imagenes aleatoriamente
        layers.RandomFlip('horizontal'),
        # Rota imagenes hasta 15 grados
        layers.RandomRotation(15 / 360),
        # Desplaza imagenes 10% en X e Y
        layers.RandomTranslation(0.1, 0.1),
        # Escala imagenes entre -10% y +10%
        layers.RandomZoom((-0.1, 0.1)),
        # Ajusta brillo +-20%
        layers.RandomBrightness(0.2),
        # Ajusta contraste +-20%
        layers.RandomContrast(0.2),
    ], name='augmentation')


# Funcion wrapper para aplicar augmentation solo en modo entrenamiento
def _apply_augmentation(augmentation_layer):
    # Retorna una funcion que aplica augmentation
    def apply(image, label):
        # training=True asegura que se apliquen las transformaciones aleatorias
        image = augmentation_layer(image, training=True)
        return image, label
    return apply


# Construye los dataloaders para train, val y test
def build_loaders(use_augmentation: bool = False, batch_size: int = BATCH_SIZE):
    # Dataset de entrenamiento (95% del TRAIN_PATH)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode='int',
        validation_split=0.05,
        subset='training',
        seed=42,
    )
    # Dataset de validacion (5% del TRAIN_PATH)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode='int',
        validation_split=0.05,
        subset='validation',
        seed=42,
    )
    # Dataset de test
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEST_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode='int',
    )

    # No normalizamos aca: el modelo incluye Rescaling(1/255)

    # Si se pide, aplica augmentation solo a train_ds
    if use_augmentation:
        augmentation = _build_augmentation()
        train_ds = train_ds.map(
            _apply_augmentation(augmentation),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Optimizaciones de rendimiento
    # shuffle: mezcla ejemplos para que el modelo no vea orden similar
    # prefetch: carga datos en paralelo mientras el modelo entrena
    train_ds = train_ds.shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

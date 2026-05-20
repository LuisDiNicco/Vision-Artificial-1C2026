import tensorflow as tf
from tensorflow import keras
from keras import layers

from .tp3_config import BATCH_SIZE, IMG_SIZE, TEST_PATH, TRAIN_PATH


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant(IMAGENET_MEAN)
    std = tf.constant(IMAGENET_STD)
    image = (image - mean) / std
    return image, label


def _build_augmentation():
    return keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(15 / 360),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom((-0.1, 0.1)),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name='augmentation')


def _apply_augmentation(augmentation_layer):
    def apply(image, label):
        image = augmentation_layer(image, training=True)
        return image, label
    return apply


def build_loaders(use_augmentation: bool = False, batch_size: int = BATCH_SIZE):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode='int',
        validation_split=0.05,
        subset='training',
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode='int',
        validation_split=0.05,
        subset='validation',
        seed=42,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEST_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode='int',
    )

    train_ds = train_ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)

    if use_augmentation:
        augmentation = _build_augmentation()
        train_ds = train_ds.map(
            _apply_augmentation(augmentation),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = train_ds.shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

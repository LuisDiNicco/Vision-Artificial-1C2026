import tensorflow as tf
from tensorflow import keras
from keras import layers

from utils.tp3_config import IMG_SIZE, NUM_CLASSES


def _conv_block(filters):
    return [
        layers.Conv2D(filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
    ]


def ModeloBase(num_classes=NUM_CLASSES):
    return keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        *_conv_block(32),
        *_conv_block(64),
        *_conv_block(128),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes),
    ], name='modelo_base')


def ModeloOptimizado(num_classes=NUM_CLASSES):
    return keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        *_conv_block(32),
        *_conv_block(64),
        *_conv_block(128),
        *_conv_block(256),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes),
    ], name='modelo_optimizado')

import tensorflow as tf
from tensorflow import keras
from keras import layers

from utils.tp3_config import IMG_SIZE, NUM_CLASSES


# Bloque convolucional basico: Conv2D + MaxPooling (reduce spatial size)
def _conv_block(filters):
    return [
        layers.Conv2D(filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
    ]


# Modelo base con 3 bloques convolucionales para comparacion
# Arquitectura: Conv32 -> Conv64 -> Conv128 -> Dense64 -> Output
def ModeloBase(num_classes=NUM_CLASSES):
    return keras.Sequential([
        # Input: imagen 150x150 RGB
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        # 3 bloques convolucionales con incremento de filtros
        *_conv_block(32),
        *_conv_block(64),
        *_conv_block(128),
        # Reduce caracteristicas a un vector 1D
        layers.GlobalAveragePooling2D(),
        # Capas densas para clasificacion
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes),
    ], name='modelo_base')


# Modelo optimizado con 4 bloques convolucionales y mas parametros
# Arquitectura: Conv32 -> Conv64 -> Conv128 -> Conv256 -> Dense128 -> Output
# Mejoras: mas capas convolucionales y mas neuronas en capas densas
def ModeloOptimizado(num_classes=NUM_CLASSES):
    return keras.Sequential([
        # Input: imagen 150x150 RGB
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        # 4 bloques convolucionales (uno mas que modelo base)
        *_conv_block(32),
        *_conv_block(64),
        *_conv_block(128),
        *_conv_block(256),  # Bloque adicional para mas capacidad
        # Reduce caracteristicas a un vector 1D
        layers.GlobalAveragePooling2D(),
        # Capas densas mas grandes
        layers.Dense(128, activation='relu'),  # 128 neuronas (vs 64 en base)
        layers.Dense(num_classes),
    ], name='modelo_optimizado')

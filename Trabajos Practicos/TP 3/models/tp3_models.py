import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers

from utils.tp3_config import IMG_SIZE, NUM_CLASSES

# Optimizaciones usadas en ambos modelos:
# - Rescaling: normaliza entradas a [0,1]
# - BatchNormalization: estabiliza el entrenamiento
# - Dropout: reduce overfitting
# - L2: penaliza pesos grandes (regularizacion)

# Regularizacion L2 suave para evitar overfitting
L2_FACTOR = 1e-4


# Bloque convolucional basico: Conv2D + BatchNorm + MaxPooling
# - Conv2D extrae features
# - BatchNorm estabiliza activaciones
# - MaxPooling reduce tamaño espacial
def _conv_block(filters):
    return [
        layers.Conv2D(
            filters,
            3,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(L2_FACTOR),
        ),
        # Normaliza activaciones para entrenamiento mas estable
        layers.BatchNormalization(),
        # Reduce a la mitad el tamaño (downsampling)
        layers.MaxPooling2D(2),
    ]


    # Modelo base (menos capacidad, menor riesgo de overfitting)
    # Arquitectura: Conv32 -> Conv64 -> Conv128 -> Dense64 -> Output
    # Diferencia clave: 3 bloques conv y densa mas chica
def ModeloBase(num_classes=NUM_CLASSES):
    return keras.Sequential([
        # Input: imagen 150x150 RGB
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        # Normaliza a [0,1] dentro del modelo
        layers.Rescaling(1.0 / 255),
        # 3 bloques convolucionales con incremento de filtros
        *_conv_block(32),
        *_conv_block(64),
        *_conv_block(128),
        # Reduce caracteristicas a un vector 1D
        layers.GlobalAveragePooling2D(),
        # Capas densas para clasificacion
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_FACTOR)),
        # Dropout leve para reducir overfitting
        layers.Dropout(0.2),
        # Capa final (logits) para clasificacion multiclase
        layers.Dense(num_classes, kernel_regularizer=regularizers.l2(L2_FACTOR)),
    ], name='modelo_base')


    # Modelo optimizado (mas capacidad + regularizacion)
    # Arquitectura: Conv32 -> Conv64 -> Conv128 -> Conv256 -> Dense128 -> Output
    # Diferencias clave vs base:
    # - 1 bloque conv extra (256 filtros)
    # - Densa mas grande (128 vs 64)
    # - Dropout mas alto (0.3 vs 0.2)
def ModeloOptimizado(num_classes=NUM_CLASSES):
    return keras.Sequential([
        # Input: imagen 150x150 RGB
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        # Normaliza a [0,1] dentro del modelo
        layers.Rescaling(1.0 / 255),
        # 4 bloques convolucionales (uno mas que modelo base)
        *_conv_block(32),
        *_conv_block(64),
        *_conv_block(128),
        *_conv_block(256),  # Bloque adicional para mas capacidad
        # Reduce caracteristicas a un vector 1D
        layers.GlobalAveragePooling2D(),
        # Capas densas mas grandes
        layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(L2_FACTOR),
        ),  # 128 neuronas (vs 64 en base)
        # Dropout mas alto por tener mas capacidad
        layers.Dropout(0.3),
        # Capa final (logits) para clasificacion multiclase
        layers.Dense(num_classes, kernel_regularizer=regularizers.l2(L2_FACTOR)),
    ], name='modelo_optimizado')

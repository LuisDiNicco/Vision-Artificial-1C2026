import tensorflow as tf

# Detecta si hay GPU disponible y retorna el dispositivo a usar
def get_device():
    # Obtiene lista de GPUs disponibles
    gpus = tf.config.list_physical_devices('GPU')
    # Si hay GPU, retorna su nombre
    if gpus:
        return f"GPU: {gpus[0].name}"
    # Sino, usa CPU
    return "CPU"


# Configura TensorFlow para usar la GPU de manera eficiente
def configure_tensorflow():
    # Obtiene lista de GPUs
    gpus = tf.config.list_physical_devices('GPU')
    # Para cada GPU, activa memory growth (no consume toda memoria al inicio)
    for gpu in gpus:
        try:
            # Evita que TensorFlow reserve toda la VRAM de una vez
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Si hay error, continua sin memoria dinamica
            pass


# Retorna la estrategia de distribucion para entrenar el modelo
def get_strategy():
    # Obtiene lista de GPUs
    gpus = tf.config.list_physical_devices('GPU')
    # Si hay GPU, usa estrategia de GPU unica
    if gpus:
        return tf.distribute.OneDeviceStrategy("/gpu:0")
    # Sino, usa estrategia por defecto (CPU)
    return tf.distribute.get_strategy()

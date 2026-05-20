import tensorflow as tf


def get_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return f"GPU: {gpus[0].name}"
    return "CPU"


def configure_tensorflow():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def get_strategy():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return tf.distribute.OneDeviceStrategy("/gpu:0")
    return tf.distribute.get_strategy()

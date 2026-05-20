from pathlib import Path

from .runtime.device_utils import configure_tensorflow, get_device

# Configuracion del proyecto TP 3
# ================================
# Este archivo centraliza todas las rutas y parametros de entrenamiento

# Directorio raiz del TP (donde estan entrenador.py y evaluador.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas del dataset
DATASET_PATH = BASE_DIR / "Imaganes de Paisajes"
TRAIN_PATH = DATASET_PATH / "seg_train" / "seg_train"
TEST_PATH = DATASET_PATH / "seg_test" / "seg_test"
# Directorios donde guardar resultados y salidas
TRAIN_OUTPUT_DIR = BASE_DIR / "salida entrenamiento"
EVAL_OUTPUT_DIR = BASE_DIR / "salida evaluacion"

# Hiperparametros del modelo
IMG_SIZE = 150  # Resolucion de entrada
NUM_CLASSES = 6  # Cantidad de clases (buildings, forest, glacier, mountain, sea, street)
BATCH_SIZE = 32  # Imagenes por batch en entrenamiento
EPOCHS = 50  # Epocas de entrenamiento
LEARNING_RATE = 1e-3  # Tasa de aprendizaje SGD

# Nombres de clases en ingles y espanol
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_NAMES_ES = ["edificios", "bosques", "glaciares", "montanas", "mar", "calles"]

# Rutas donde guardar los modelos entrenados
MODEL_PATHS = {
    "base": BASE_DIR / "modelos_guardados" / "modelo_base.keras",
    "augmentation": BASE_DIR / "modelos_guardados" / "modelo_augmentation.keras",
    "optimized": BASE_DIR / "modelos_guardados" / "modelo_optimizado.keras",
}

# Detectar dispositivo (GPU o CPU) y configurar TensorFlow
DEVICE = get_device()
configure_tensorflow()

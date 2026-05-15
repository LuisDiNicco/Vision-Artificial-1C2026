from pathlib import Path

from .runtime.device_utils import configure_torch, get_device, get_num_workers, use_pin_memory


BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas del dataset y carpetas de salida.
DATASET_PATH = BASE_DIR / "Imaganes de Paisajes"
TRAIN_PATH = DATASET_PATH / "seg_train" / "seg_train"
TEST_PATH = DATASET_PATH / "seg_test" / "seg_test"
TRAIN_OUTPUT_DIR = BASE_DIR / "salida entrenamiento"
EVAL_OUTPUT_DIR = BASE_DIR / "salida evaluacion"

# Hiperparametros basicos del entrenamiento.
IMG_SIZE = 150
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

# Nombres de clases en ingles y espanol para mostrar en reportes.
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_NAMES_ES = ["edificios", "bosques", "glaciares", "montanas", "mar", "calles"]

# Rutas donde se guardan los modelos entrenados.
MODEL_PATHS = {
    "base": BASE_DIR / "modelos_guardados" / "modelo_base.pt",
    "augmentation": BASE_DIR / "modelos_guardados" / "modelo_augmentation.pt",
    "optimized": BASE_DIR / "modelos_guardados" / "modelo_optimizado.pt",
}

# Dispositivo de ejecucion: CPU o GPU.
DEVICE = get_device()
configure_torch(DEVICE)
USE_PIN_MEMORY = use_pin_memory(DEVICE)
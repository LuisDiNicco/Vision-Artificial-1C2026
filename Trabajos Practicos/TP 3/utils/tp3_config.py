from pathlib import Path

from .runtime.device_utils import configure_tensorflow, get_device


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "Imaganes de Paisajes"
TRAIN_PATH = DATASET_PATH / "seg_train" / "seg_train"
TEST_PATH = DATASET_PATH / "seg_test" / "seg_test"
TRAIN_OUTPUT_DIR = BASE_DIR / "salida entrenamiento"
EVAL_OUTPUT_DIR = BASE_DIR / "salida evaluacion"

IMG_SIZE = 150
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_NAMES_ES = ["edificios", "bosques", "glaciares", "montanas", "mar", "calles"]

MODEL_PATHS = {
    "base": BASE_DIR / "modelos_guardados" / "modelo_base.keras",
    "augmentation": BASE_DIR / "modelos_guardados" / "modelo_augmentation.keras",
    "optimized": BASE_DIR / "modelos_guardados" / "modelo_optimizado.keras",
}

DEVICE = get_device()
configure_tensorflow()

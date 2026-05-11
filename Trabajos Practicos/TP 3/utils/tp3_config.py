from pathlib import Path
import os

import torch


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "Imaganes de Paisajes"
TRAIN_PATH = DATASET_PATH / "seg_train" / "seg_train"
TEST_PATH = DATASET_PATH / "seg_test" / "seg_test"
TRAIN_OUTPUT_DIR = BASE_DIR / "salida entrenamiento"
EVAL_OUTPUT_DIR = BASE_DIR / "salida evaluacion"

IMG_SIZE = 150
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_NAMES_ES = ["edificios", "bosques", "glaciares", "montanas", "mar", "calles"]

MODEL_PATHS = {
    "base": BASE_DIR / "modelos_guardados" / "modelo_base.pt",
    "augmentation": BASE_DIR / "modelos_guardados" / "modelo_augmentation.pt",
    "optimized": BASE_DIR / "modelos_guardados" / "modelo_optimizado.pt",
}

# Detectar GPU disponible (NVIDIA CUDA, AMD DirectML, o CPU)
def _get_device():
    # Intenta CUDA primero (NVIDIA)
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", "NVIDIA (CUDA)"

    # Intenta DirectML (AMD en Windows)
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device(), "dml", "AMD (DirectML)"
    except ImportError:
        pass
    except Exception:
        pass

    # Intenta Metal (macOS)
    try:
        if hasattr(torch, "mps") and torch.mps.is_available():
            return torch.device("mps"), "mps", "macOS (Metal)"
    except Exception:
        pass

    # Fallback a CPU
    return torch.device("cpu"), "cpu", "CPU"


DEVICE, DEVICE_TYPE, GPU_TYPE = _get_device()
USE_AMP = DEVICE_TYPE == "cuda"  # AMP solo funciona con CUDA

if DEVICE_TYPE == "cuda":
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def get_num_workers() -> int:
    cpu_count = os.cpu_count() or 2
    if os.name == "nt":
        return max(1, min(2, cpu_count - 1))
    return max(1, min(4, cpu_count - 1))
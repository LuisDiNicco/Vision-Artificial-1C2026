import os
import warnings

import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except Exception:
        pass

    try:
        if hasattr(torch, "mps") and torch.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass

    return torch.device("cpu")


def configure_torch(device):
    if device.type != "cuda":
        return
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def use_pin_memory(device):
    return device.type == "cuda"


def get_num_workers():
    cpu_count = os.cpu_count() or 2
    if os.name == "nt":
        return max(1, min(2, cpu_count - 1))
    return max(1, min(4, cpu_count - 1))


def safe_torch_load(model_path, device):
    # Cargamos el modelo de forma compatible con distintas versiones de PyTorch.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            return torch.load(model_path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError, Exception):
            return torch.load(model_path, map_location=device, weights_only=False)

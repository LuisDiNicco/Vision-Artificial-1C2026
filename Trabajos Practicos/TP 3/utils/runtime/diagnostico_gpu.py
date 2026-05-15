"""
Diagnostico simple del dispositivo de ejecucion.
"""
import platform
import sys

import torch

from utils.runtime.device_utils import get_device


print("=" * 60)
print("DIAGNOSTICO DE DISPOSITIVO - TP 3")
print("=" * 60)

print(f"\nSistema Operativo: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")

print("\n" + "=" * 60)
print("DETECCION DEL DISPOSITIVO")
print("=" * 60)

print(f"\nCUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

if hasattr(torch, "mps"):
    try:
        print(f"MPS disponible: {torch.mps.is_available()}")
    except Exception as exc:
        print(f"MPS disponible: Error ({exc})")

print("\n" + "=" * 60)
print("CONFIGURACION DEL PROYECTO")
print("=" * 60)

try:
    device = get_device()
    print(f"\nDispositivo detectado: {device}")
except Exception as exc:
    print(f"\nError al detectar dispositivo: {exc}")

print("\n" + "=" * 60)

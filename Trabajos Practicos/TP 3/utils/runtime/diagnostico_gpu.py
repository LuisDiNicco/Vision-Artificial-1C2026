"""
Diagnostico simple del dispositivo de ejecucion.
"""
import platform
import sys

import tensorflow as tf

from utils.runtime.device_utils import get_device


print("=" * 60)
print("DIAGNOSTICO DE DISPOSITIVO - TP 3")
print("=" * 60)

print(f"\nSistema Operativo: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")
print(f"TensorFlow: {tf.__version__}")

print("\n" + "=" * 60)
print("DETECCION DEL DISPOSITIVO")
print("=" * 60)

gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU disponible: {len(gpus) > 0}")
if gpus:
    for gpu in gpus:
        print(f"  - {gpu.name}")

cpus = tf.config.list_physical_devices('CPU')
print(f"CPUs detectadas: {len(cpus)}")

print("\n" + "=" * 60)
print("CONFIGURACION DEL PROYECTO")
print("=" * 60)

try:
    device = get_device()
    print(f"\nDispositivo detectado: {device}")
except Exception as exc:
    print(f"\nError al detectar dispositivo: {exc}")

print("\n" + "=" * 60)

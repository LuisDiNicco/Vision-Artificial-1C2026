"""
Script de diagnóstico para verificar GPU AMD y DirectML
"""
import torch
import platform
import sys

print("=" * 60)
print("DIAGNÓSTICO DE GPU - TP 3")
print("=" * 60)

print(f"\n📊 Sistema Operativo: {platform.system()} {platform.release()}")
print(f"🐍 Python: {sys.version.split()[0]}")
print(f"🔥 PyTorch: {torch.__version__}")

print("\n" + "=" * 60)
print("DETECCIÓN DE HARDWARE")
print("=" * 60)

# CUDA (NVIDIA)
print(f"\n✓ CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - Capability: {torch.cuda.get_device_capability(0)}")

# DirectML (AMD en Windows)
try:
    import torch_directml
    print(f"\n✓ torch_directml disponible: True")
    print(f"  - dml.is_available(): {torch_directml.is_available()}")
except ImportError:
    print(f"\n✓ torch_directml disponible: False")
except Exception as e:
    print(f"\n✓ torch_directml disponible: Error\n  - Error: {e}")

# Metal (macOS)
print(f"\n✓ torch.mps disponible: {hasattr(torch, 'mps')}")
if hasattr(torch, 'mps'):
    try:
        print(f"  - mps.is_available(): {torch.mps.is_available()}")
    except Exception as e:
        print(f"  - Error: {e}")

print("\n" + "=" * 60)
print("CONFIGURACIÓN DEL PROYECTO")
print("=" * 60)

try:
    from utils.tp3_config import DEVICE, DEVICE_TYPE, GPU_TYPE
    print(f"\n✓ Dispositivo detectado: {DEVICE}")
    print(f"  - Tipo: {DEVICE_TYPE}")
    print(f"  - Descripción: {GPU_TYPE}")
except Exception as e:
    print(f"\n❌ Error al importar config: {e}")

print("\n" + "=" * 60)
print("RECOMENDACIÓN")
print("=" * 60)

has_dml = False
try:
    import torch_directml
    has_dml = torch_directml.is_available()
except ImportError:
    pass

if not torch.cuda.is_available() and not has_dml:
    print("""
⚠️  NO SE DETECTÓ GPU AMD CON DIRECTML

SOLUCIONES:

1️⃣  Reinstalar PyTorch con una versión estable compatible con tu Python:
    
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    (Si tu Python es 3.14, cambia a Python 3.13 o 3.12 en un entorno virtual)

2️⃣  Si todavía aparece el error, instalar Python 3.13 o 3.12 y repetir:
    
    pip install torch torchvision torchaudio

3️⃣  Si nada funciona, seguir usando CPU (es más lento pero funciona):
    
    python entrenador.py --modo base

⚠️  IMPORTANTE: Para AMD, el paquete es torch-directml y se importa como torch_directml.
""")
else:
    print(f"\n✅ GPU detectada correctamente: {GPU_TYPE if 'GPU_TYPE' in locals() else 'DirectML/CUDA'}")
    print("   Puedes ejecutar: python entrenador.py --modo todos")

print("\n" + "=" * 60)
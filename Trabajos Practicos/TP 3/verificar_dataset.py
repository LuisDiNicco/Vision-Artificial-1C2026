"""Script para verificar la estructura del dataset"""
import os
from pathlib import Path

print("\n📁 VERIFICACIÓN DE ESTRUCTURA DEL DATASET\n")

# Directorio actual
print(f"Directorio actual: {Path('.').resolve()}\n")

# Buscar carpetas de imágenes
for root, dirs, files in os.walk("."):
    nivel = root.count(os.sep)
    indent = " " * 2 * nivel
    print(f"{indent}{os.path.basename(root)}/")
    
    # Mostrar primeros archivos
    subindent = " " * 2 * (nivel + 1)
    for file in files[:3]:
        print(f"{subindent}{file}")
    
    if len(files) > 3:
        print(f"{subindent}... y {len(files)-3} archivos más")
    
    # Limitar profundidad
    if nivel > 3:
        break

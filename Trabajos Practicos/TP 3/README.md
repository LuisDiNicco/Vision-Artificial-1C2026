# TP 3 - Clasificación de Paisajes con CNN

## ¿Qué pide la consigna?

La consigna solicita entrenar un modelo de inteligencia artificial (clasificador, segmentador, etc.) usando un dataset a elección y mostrar:

1. **Ejecución del modelo con imágenes de prueba** → demostrar que el modelo funciona haciendo predicciones
2. **Métricas obtenidas durante el entrenamiento** → mostrar cómo evolucionan la precisión y la pérdida a lo largo de los epochs
3. **Técnicas utilizadas para mejorar las métricas** → explicar qué se hizo para que el modelo sea más preciso

---

## Solución Implementada

### Objetivo
Clasificar imágenes de paisajes en 6 categorías: **buildings, forest, glacier, mountain, sea, street** usando el Intel Image Classification Dataset.

### Arquitectura
Se implementaron **3 modelos con diferentes niveles de sofisticación**:

#### 1. **Modelo Base**
- Arquitectura simple: 3 bloques convolucionales (32 → 64 → 128 canales)
- Sin regularización extra
- Clasificador simple: 2 capas densas (128 → 64 → 6 clases)
- Rápido pero menos preciso

#### 2. **Modelo con Augmentation (Data)**
- Mismo modelo base
- Pero entrena con técnicas de aumento de datos:
  - Rotaciones aleatorias (±15°)
  - Flips horizontales (50% de probabilidad)
  - Transformaciones afines (escala y traslación)
- Mejora la generalización sin cambiar la arquitectura

#### 3. **Modelo Optimizado**
- Arquitectura mejorada: 4 bloques convolucionales (32 → 64 → 128 → 256 canales)
- **Batch Normalization** en cada bloque (estabiliza el entrenamiento)
- **Dropout** progresivo (10% → 15% → 20% → 25%) para evitar overfitting
- **Learning Rate Scheduler** (ReduceLROnPlateau) para ajustar el aprendizaje automáticamente
- Alcanza mejor precisión y generalización

---

## Técnicas para Mejorar las Métricas

### 1. **Data Augmentation**
Aumenta artificialmente el dataset mediante transformaciones de las imágenes de entrenamiento:
- Rotaciones, flips y traslaciones aleatorias
- Hace que el modelo sea más robusto ante variaciones

### 2. **Batch Normalization**
Normaliza las activaciones entre capas:
- Estabiliza el entrenamiento
- Permite usar learning rates más altos
- Mejora la convergencia

### 3. **Dropout**
Desactiva aleatoriamente neuronas durante el entrenamiento:
- Previene overfitting (memorizar datos específicos)
- Fuerza al modelo a aprender características más generales
- Aumenta de 10% → 25% en capas más profundas

### 4. **Optimizador AdamW**
Versión mejorada de Adam:
- Incluye weight decay (regularización L2)
- Evita que los pesos crezcan demasiado

### 5. **Learning Rate Scheduler (ReduceLROnPlateau)**
Ajusta dinámicamente la tasa de aprendizaje:
- Si la pérdida de validación no mejora, reduce el learning rate
- Permite fine-tuning en etapas finales del entrenamiento

### 6. **Early Stopping**
Detiene el entrenamiento después de N epochs sin mejora:
- Evita que el modelo se sobreentrenadiente
- Guarda automáticamente el mejor modelo

### 7. **Mixed Precision (AMP)**
Entrena con precisión mixta (float16/float32):
- Acelera el entrenamiento en GPU
- Reduce el uso de memoria

### 8. **AdaptiveAvgPool2d**
Capa de pooling adaptativo:
- Adapta automáticamente cualquier tamaño de entrada a una salida fija
- Evita capas densas enormes (mucho más eficiente)

---

## Estructura del Dataset

El dataset **Intel Image Classification** está organizado de la siguiente manera:

```
Imaganes de Paisajes/
├── seg_train/                    (Imágenes de entrenamiento)
│   ├── buildings/                ~ 2000 imágenes
│   ├── forest/                   ~ 2000 imágenes
│   ├── glacier/                  ~ 2000 imágenes
│   ├── mountain/                 ~ 2000 imágenes
│   ├── sea/                      ~ 2000 imágenes
│   └── street/                   ~ 2000 imágenes
│
└── seg_test/                     (Imágenes de prueba)
    ├── buildings/                ~ 260 imágenes
    ├── forest/                   ~ 260 imágenes
    ├── glacier/                  ~ 260 imágenes
    ├── mountain/                 ~ 260 imágenes
    ├── sea/                      ~ 260 imágenes
    └── street/                   ~ 260 imágenes
```

**Total:** ~14,600 imágenes (12,800 de entrenamiento + 1,800 de prueba)

Cada imagen es un archivo `.jpg` de 150×150 píxeles. Las carpetas están organizadas por clase, lo que permite a `torchvision.datasets.ImageFolder` cargarlas automáticamente.

---

## Cómo Ejecutar

### Requisitos previos

**Importante:** PyTorch suele publicar ruedas para Python 3.12/3.13 antes que para versiones nuevas. Si estás usando Python 3.14 y ves `No matching distribution found for torch`, instala Python 3.13 o 3.12 en un entorno virtual antes de seguir.

**Instalación base (CPU):**
```bash
pip install torch torchvision torchaudio matplotlib scikit-learn
```

**Para GPU NVIDIA (CUDA):**
```bash
# Windows/Linux - reemplaza cu118 por tu versión de CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Para GPU AMD en Windows (DirectML - Recomendado):**

**⚠️ IMPORTANTE:** `torch-directml` como paquete separado ya NO EXISTE. DirectML está integrado en PyTorch 2.0+.

```bash
# Opción 1: Instalar una versión estable compatible con tu Python
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Opción 2: Si tu Python sigue sin tener rueda compatible, bajar a Python 3.13 o 3.12
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio
```

> Luego ejecuta `python diagnostico_gpu.py` para verificar que DirectML se detectó correctamente

### Entrenar los modelos

**Verificar GPU AMD primero** (solo si tienes GPU AMD):
```bash
python diagnostico_gpu.py
```

**Entrenar todos los modelos** (toma ~30-50 minutos con GPU):
```bash
python entrenador.py
# O equivalente:
python entrenador.py --modo todos
```

**Entrenar un modelo específico:**
```bash
python entrenador.py --modo base          # Solo modelo base
python entrenador.py --modo augmentation  # Solo con augmentation
python entrenador.py --modo optimized     # Solo modelo optimizado
```

El script:
- Entrena el modelo en GPU si está disponible, sino usa CPU
- Guarda el mejor modelo en `modelo_base.pt`, `modelo_augmentation.pt`, `modelo_optimizado.pt`
- Genera gráficos comparativos en `comparacion_resultados.png`
- Guarda los resultados en `resultados_entrenamiento.json`

### Evaluar los modelos
```bash
python evaluador.py
```

Genera:
- **Matriz de confusión** (`matriz_confusion.png`) - muestra qué clases se confunden
- **Predicciones aleatorias** (`predicciones_aleatorias.png`) - 9 ejemplos del modelo predicitendo
- **Métricas por clase** (Precision, Recall, F1-Score)
- **Accuracy general** del modelo

### Ejemplo de salida esperada
```
Entrenando modelo_base
Dispositivo: cuda
AMP: si

Epoch 1/30 - Loss: 1.45 | Acc: 0.52 | Val Loss: 1.22 | Val Acc: 0.60
Epoch 2/30 - Loss: 1.10 | Acc: 0.64 | Val Loss: 1.00 | Val Acc: 0.68
...
Epoch 30/30 - Loss: 0.45 | Acc: 0.88 | Val Loss: 0.55 | Val Acc: 0.84

Test Accuracy: 0.84
```

---

## Archivos Principales

| Archivo | Ubicación | Función |
|---------|-----------|----------|
| `entrenador.py` | Raíz | Script principal para entrenar modelos |
| `evaluador.py` | Raíz | Script para evaluar un modelo guardado |
| `tp3_models.py` | `models/` | Define la arquitectura del modelo base y optimizado |
| `tp3_data.py` | `utils/` | Carga datos y aplica transformaciones |
| `tp3_training.py` | `utils/` | Lógica de entrenamiento, validación y generación de gráficos |
| `tp3_config.py` | `utils/` | Rutas, constantes y configuración global |

---

## Archivos de Salida Generados

Todos se generan automáticamente al ejecutar los scripts:

**Modelos entrenados** (guardados en `modelos_guardados/`):
- `modelo_base.pt` - Mejor modelo base entrenado
- `modelo_augmentation.pt` - Mejor modelo con augmentation
- `modelo_optimizado.pt` - Mejor modelo optimizado

**Métricas y gráficos** (guardados en raíz):
- `resultados_entrenamiento.json` - Métricas de cada modelo en formato JSON
- `comparacion_resultados.png` - Gráfico comparando los 3 modelos (accuracy vs loss)
- `matriz_confusion.png` - Matriz de confusión del mejor modelo
- `predicciones_aleatorias.png` - 9 ejemplos de predicciones correctas/incorrectas
- `modelo_*_history.png` - Gráficos de accuracy y loss por epoch para cada modelo

---

## Optimizaciones Implementadas

✅ **GPU acceleration** - Soporta NVIDIA (CUDA), AMD (DirectML en Windows) y macOS (Metal)  
✅ **torch.compile()** - Compilación JIT en PyTorch 2.0+ (acelera GPU 20-30%)  
✅ **Mixed Precision (AMP)** - Entrena más rápido en GPU NVIDIA (float16/float32)  
✅ **Efficient DataLoader** - `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers`  
✅ **Early Stopping** - Evita entrenar de más (paciencia de 4 epochs)  
✅ **Learning Rate Scheduler** - Ajusta automáticamente el learning rate (ReduceLROnPlateau)  
✅ **AdaptiveAvgPool** - Reduce parámetros sin perder información  
✅ **Imports relativos** - Los módulos usan imports relativos para ser portables entre máquinas  

---

## Notas Importantes

### Hardware

- **Para GPU NVIDIA:** El código usa CUDA automáticamente (recomendado)
- **Para GPU AMD en Windows (Radeon RX 6000+):** Usa DirectML (integrado en PyTorch 2.0+)
  - Reinstala PyTorch con una versión compatible de Python (idealmente 3.13 o 3.12)
  - Verifica con: `python diagnostico_gpu.py`
  - No requiere ROCm ni configuración adicional
  - Tu driver normal de AMD Adrenalin es suficiente
- **Para macOS con GPU:** El código detecta Metal automáticamente
- **CPU:** Funciona en cualquier máquina, pero es lento (~5-10 min/epoch vs 10-30 seg/epoch en GPU)

### Otros puntos

- El dataset tiene aproximadamente 14,000 imágenes (12,800 train + 1,600 test)
- Cada epoch toma ~10-30 segundos en GPU (NVIDIA) o ~15-40 segundos en AMD DirectML
- En CPU toma ~5-10 minutos por epoch
- El modelo optimizado es más lento de entrenar pero más preciso
- Puedes interrumpir el entrenamiento con Ctrl+C sin problemas (se guarda el mejor modelo encontrado hasta ese momento)
# TP 3 - Clasificación de Paisajes con CNN

## ¿Qué pide la consigna?

La consigna solicita entrenar un modelo de inteligencia artificial (clasificador, segmentador, etc.) usando un dataset a elección y mostrar:

1. **Ejecución del modelo con imágenes de prueba** → demostrar que el modelo funciona haciendo predicciones
2. **Métricas obtenidas durante el entrenamiento** → mostrar cómo evolucionan la precisión y la pérdida a lo largo de los epochs
3. **Técnicas utilizadas para mejorar las métricas** → explicar qué se hizo para que el modelo sea más preciso

---

## Solución Implementada

### Objetivo
Clasificar imágenes de paisajes en 6 categorías: **buildings, forest, glacier, mountain, sea, street** usando el Intel Image Classification Dataset. El trabajo se enfoca en cumplir la consigna con un modelo entrenable, una evaluación clara y una comparación simple entre variantes.

### Arquitectura
Se implementaron **3 variantes del mismo problema** para comparar resultados:

#### 1. **Modelo Base**
- Es la versión más simple.
- Sirve como línea base para comparar.

#### 2. **Modelo con Augmentation (Data)**
- Mantiene la misma estructura del modelo base.
- Cambia la forma de entrenar usando imágenes modificadas levemente para que el modelo vea más ejemplos.
- Ayuda a que generalice mejor.
- Aumentaciones usadas: flip horizontal, rotación leve, traslación y zoom.

#### 3. **Modelo Optimizado**
- Es la versión más completa.
- Usa una red más profunda (más capas convolucionales) y está pensada para dar mejores resultados que las dos anteriores.
- Permite ver si una arquitectura más fuerte mejora la evaluación final.

---

## Qué se hizo para cumplir la consigna

Se armó un flujo completo: cargar el dataset, entrenar varios modelos, guardar el mejor resultado, evaluar sobre el conjunto de prueba y generar archivos para mostrar el desempeño.

**Justificación de decisiones**
- **Tipo de modelo**: se eligió una CNN porque extrae características visuales con capas convolucionales y luego clasifica con capas totalmente conectadas, que es la estructura estándar para clasificación de imágenes.
- **Preprocesamiento**: todas las imágenes se redimensionan a un tamaño fijo y se normalizan por canal para estabilizar el entrenamiento y mantener la escala de los valores.
- **Separación de datos**: se divide el conjunto de entrenamiento en entrenamiento (95%) y validación (5%), y el conjunto de prueba queda aparte para evaluar al final.
- **Entrenamiento**: se usa una función de pérdida de clasificación multiclase y optimización con SGD, registrando accuracy y loss por época para ver la evolución.
- **Criterio de corte**: se aplica early stopping cuando la pérdida de validación deja de mejorar para evitar sobreentrenamiento innecesario.

**Técnicas para mejorar métricas**
- **Aumentación**: se generan variantes de las imágenes (flip, rotación, traslación, zoom, brillo y contraste) para aumentar la diversidad del dataset y mejorar la generalización.
- **Modelo más profundo**: la versión optimizada agrega más bloques convolucionales para capturar patrones de mayor nivel.
- **SGD con momentum**: agrega inercia al ajuste de pesos para estabilizar el descenso del error.

En la evaluación se generan la matriz de confusión, las métricas por clase y ejemplos de predicciones para mostrar si el modelo aprendió a distinguir las categorías.

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

### Configuración del entorno (desde cero, Python 3.12 o inferior)

1) Abrí una terminal en la carpeta del TP 3:
```bash
cd "F:\Luis\UNLAM\17º UNLAM DECIMO-TERCER CUATRIMESTRE (1C2026)\Vision Artificial\Vision-Artificial-1C2026\Trabajos Practicos\TP 3"
```

2) Creá el entorno virtual con Python 3.12:
```bash
py -3.12 -m venv .venv
# Si el comando anterior no existe, usá:
python -m venv .venv
```

3) Activá el entorno virtual:
```bash
# PowerShell
\.venv\Scripts\Activate.ps1

# CMD
\.venv\Scripts\activate

# Git Bash
source .venv/Scripts/activate
```

4) Actualizá pip:
```bash
python -m pip install --upgrade pip
```

5) Instalá dependencias:
```bash
pip install torch torchvision torchaudio matplotlib scikit-learn
```

6) Verificá que todo esté bien:
```bash
python -c "import torch; print(torch.__version__)"
```

7) Dataset: descargá el **Intel Image Classification Dataset** y asegurate de que quede así:
```
Imaganes de Paisajes/
├── seg_train/
└── seg_test/
```

**Notas rápidas**
- Si PowerShell bloquea la activación, ejecutá una vez: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
- Con Python 3.14, varias dependencias (por ejemplo PyTorch) no tienen ruedas disponibles. Instalá Python 3.12 o 3.11 y repetí los pasos.

### Requisitos previos

**Version de Python recomendada:** 3.12 (o 3.11)

**Instalación base:**
```bash
pip install torch torchvision torchaudio matplotlib scikit-learn
```

### Entrenar los modelos

**Entrenar todos los modelos**:
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
- Guarda el mejor modelo en `modelo_base.pt`, `modelo_augmentation.pt`, `modelo_optimizado.pt`
- Genera gráficos y métricas en `salida entrenamiento/`
- Guarda el resumen general en `salida entrenamiento/resultados_entrenamiento.json`
- Usa early stopping como criterio simple de terminación si la pérdida de validación deja de mejorar

### Evaluar los modelos
```bash
python evaluador.py
```

Genera:
- **Matriz de confusión** (`salida evaluacion/matriz_confusion.png`) - muestra qué clases se confunden
- **Predicciones aleatorias** (`salida evaluacion/predicciones_aleatorias.png`) - ejemplos del modelo prediciendo
- **Métricas por clase** (Precision, Recall, F1-Score)
- **Accuracy general** del modelo

### Ejemplo de salida esperada
```
Entrenando modelo_base
Optimizador: SGD

Epoca 1/20 | train 0.5200/1.4500 | val 0.6000/1.2200 | lr 1.00e-03
Epoca 2/20 | train 0.6400/1.1000 | val 0.6800/1.0000 | lr 1.00e-03
...
Epoca 20/20 | train 0.8800/0.4500 | val 0.8400/0.5500 | lr 1.00e-03

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
| `device_utils.py` | `utils/runtime/` | Utilidades de ejecución (dispositivo y carga segura) |

---

## Archivos de Salida Generados

Todos se generan automáticamente al ejecutar los scripts:

**Modelos entrenados** (guardados en `modelos_guardados/`):
- `modelo_base.pt` - Mejor modelo base entrenado
- `modelo_augmentation.pt` - Mejor modelo con augmentation
- `modelo_optimizado.pt` - Mejor modelo optimizado

**Métricas y gráficos**:
- `salida entrenamiento/resultados_entrenamiento.json` - Métricas de cada modelo en formato JSON
- `salida entrenamiento/comparacion_resultados.png` - Gráfico comparando los 3 modelos (accuracy vs loss)
- `salida entrenamiento/modelo_*_history.png` - Gráficos de accuracy y loss por epoch para cada modelo
- `salida entrenamiento/modelo_*_history.json` - Historial numérico de cada entrenamiento
- `salida evaluacion/matriz_confusion.png` - Matriz de confusión del modelo evaluado
- `salida evaluacion/predicciones_aleatorias.png` - Ejemplos de predicciones correctas/incorrectas

---

## Resumen del Flujo

1. Se carga el dataset de entrenamiento y prueba.
2. Se entrenan tres variantes del modelo para comparar resultados.
3. Se guardan los pesos del mejor modelo en `modelos_guardados/`.
4. Se generan archivos de salida separados para entrenamiento y evaluación.
5. Se usa esa información para explicar el desempeño del sistema frente a la consigna.

---

## Notas Importantes

- El dataset está organizado por carpetas, así que `ImageFolder` puede leerlo directamente.
- El entrenamiento guarda el mejor estado encontrado durante la ejecución.
- Los archivos de salida quedan separados en carpetas para no mezclar resultados de entrenamiento con evaluación.
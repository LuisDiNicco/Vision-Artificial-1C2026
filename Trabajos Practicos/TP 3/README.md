# TP 3 - Vision Artificial (1C 2026)

## Datos del trabajo practico
- Materia: Vision Artificial
- Institucion: UNLaM
- Cuatrimestre: 1C 2026

## Integrantes (Grupo 4)

| DNI | Apellido, Nombre |
|---:|---|
| 43.630.151 | Antonioli, Iván Oscar |
| 43.664.669 | Di Nicco, Luis Demetrio |
| 41.069.597 | Rojas, Tomas Ian |

## Consigna
Utilizando lo visto en esta materia y lo aprendido en materias de inteligencia artificial, entrenar un modelo (ya sea un clasificador, segmentador, etc) utilizando un dataset a elección.

Se deberá mostrar:

- Ejecución del modelo con imágenes de prueba
- Métricas obtenidas durante el entrenamiento (cambios en la precisión y en la pérdida)
- Técnicas utilizadas para mejorar las métricas

## Dataset utilizado
**Intel Image Classification Dataset** (Natural Scenes around the world)

- **Fuente**: https://datahack.analyticsvidhya.com
- **Autor original**: Intel
- **Tamaño total**: ~25k imágenes de 150x150 píxeles
- **Clases**: 6 categorías
  - `buildings` (0) - edificios
  - `forest` (1) - bosques
  - `glacier` (2) - glaciares
  - `mountain` (3) - montañas
  - `sea` (4) - mar
  - `street` (5) - calles
  
- **Distribución**:
  - Train: ~14k imágenes
  - Test: ~3k imágenes
  - Prediction: ~7k imágenes (sin etiquetas)

## Implementacion actual (Grupo 4)
Se desarrolló un modelo de red neuronal convolucional (CNN) para clasificar imágenes de paisajes naturales.

Enfoque: entrenar un modelo base simple, luego aplicar optimizaciones progresivas para mejorar las métricas de precisión y reducir la pérdida.

## Estructura de archivos

- `entrenador.py`: script que entrena el modelo con diferentes niveles de optimización
- `evaluador.py`: script que carga el modelo y evalúa en imágenes de prueba
- `modelo_base.h5`: modelo entrenado sin optimizaciones
- `modelo_optimizado.h5`: modelo entrenado con todas las optimizaciones
- `historico_entrenamiento.json`: métricas de precisión y pérdida durante el entrenamiento
- `Imaganes de Paisajes/`: carpeta con el dataset (train, test, prediction)

## Técnicas de optimización aplicadas

1. **Data Augmentation**: rotación, desplazamiento, zoom en imágenes de entrenamiento
2. **Batch Normalization**: normaliza activaciones entre capas
3. **Dropout**: regularización para evitar overfitting
4. **Learning Rate Scheduling**: ajusta tasa de aprendizaje dinámicamente
5. **Early Stopping**: detiene el entrenamiento si la validación no mejora

## Como ejecutar

```bash
# Entrenar el modelo (muestra resultados progresivos)
python entrenador.py

# Evaluar en imágenes de prueba
python evaluador.py
```
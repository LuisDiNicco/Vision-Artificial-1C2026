# Resumen de Modelos y Métricas - TP3

## Modelos Utilizados

### 1. Modelo Base (`modelo_base`)
- **Arquitectura**:
  - 3 bloques convolucionales (Conv2D + BatchNorm + MaxPooling).
  - Rescaling interno a [0, 1].
  - GlobalAveragePooling2D + Dense(64) + Dropout(0.2).
  - Regularizacion L2 suave.
- **Métricas finales**:
  - Mejor precisión en validación: **84.88%**
  - Mejor pérdida en validación: **0.4304**
  - Época del mejor punto: **16**

### 2. Modelo con Aumentación de Datos (`modelo_augmentation`)
- **Cambios**:
  - Se aplica **data augmentation** (rotaciones, zoom, ajustes de brillo, etc.) en el conjunto de entrenamiento.
  - Se mantiene la base con BatchNorm, Dropout y L2.
- **Métricas finales**:
  - Mejor precisión en validación: **86.73%**
  - Mejor pérdida en validación: **0.4017**
  - Época del mejor punto: **34**

### 3. Modelo Optimizado (`modelo_optimizado`)
- **Cambios**:
  - Arquitectura más profunda:
    - Se añade un cuarto bloque convolucional (256 filtros).
  - Densa mas grande (128 neuronas) + Dropout(0.3).
  - Se mantiene el uso de **data augmentation** y la regularizacion L2.
- **Métricas finales**:
  - Mejor precisión en validación: **87.73%**
  - Mejor pérdida en validación: **0.3784**
  - Época del mejor punto: **47**

---

## Variación de Métricas
- **Modelo Base**:
  - Es el mas estable, pero su mejor punto queda por debajo del optimizado.
- **Modelo con Aumentación**:
  - Mejora cuando se elige la mejor epoca, aunque la validacion es mas irregular.
- **Modelo Optimizado**:
  - Consigue el mejor valor de validacion, pero requiere mas epocas para alcanzarlo.

---

## Analisis

- Se reportan los mejores valores de validacion por modelo, tomados de los historiales.
- El optimizado logra el mejor balance (precision alta y perdida mas baja) al costo de mas epocas.
- La aumentacion alcanza un buen pico, pero la curva es inestable.
- Las metricas dependen de la semilla y del entorno; este resumen refleja la ultima corrida guardada.

---

## Cambios Adicionales Recomendados

1. **Transfer Learning**
   - Utilizar modelos preentrenados como ResNet o EfficientNet para aprovechar características ya aprendidas en grandes conjuntos de datos.

2. **Optimización de Hiperparámetros**
   - Ajustar la tasa de aprendizaje, el tamaño del batch y el número de épocas.
   - Probar optimizadores más avanzados como Adam o RMSprop.

3. **Regularización**
   - Implementar técnicas como Dropout o L2 regularization para reducir el sobreajuste.

4. **Aumentación de Datos Avanzada**
   - Incluir técnicas como recortes aleatorios, normalización por canales o mezclado de imágenes (e.g., MixUp).

5. **Arquitectura**
   - Experimentar con arquitecturas más complejas, como añadir capas residuales o bloques de atención.

6. **Validación Cruzada**
   - Dividir los datos en múltiples particiones para evaluar la robustez del modelo en diferentes subconjuntos.

7. **Evaluación Detallada**
   - Analizar la matriz de confusión para identificar clases con bajo rendimiento y ajustar el modelo o los datos en consecuencia.
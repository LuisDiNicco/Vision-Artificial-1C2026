# NOTAS DE IMPLEMENTACIÓN - TP 3

## Qué se ha creado:

### 1. **entrenador.py** - Script principal de entrenamiento
- Implementa 3 modelos CNN progresivos con PyTorch
- **Modelo 1 (BASE)**: Sin optimizaciones, arquitectura simple
  - 3 bloques conv (32, 64, 128 filtros)
  - Sin regularización
  - Útil para línea base de comparación
  
- **Modelo 2 (AUGMENTATION)**: Igual arquitectura pero con data augmentation
  - Rotación ±20°, zoom ±20%, desplazamiento ±20%
  - Flip horizontal aleatorio
  - Muestra impacto de aumentar datos artificialmente
  
- **Modelo 3 (OPTIMIZADO)**: Arquitectura mejorada con todas técnicas
  - 4 bloques conv (32, 64, 128, 256 filtros)
  - Batch Normalization en cada capa
  - Dropout (30-50%) para regularización
  - Learning Rate Scheduling (ReduceLROnPlateau)
  - Kaiming Initialization para convergencia rápida
  - Early Stopping automático

### 2. **evaluador.py** - Script para evaluar modelos
- Carga modelos entrenados (.pt)
- Realiza predicciones en imágenes individuales
- Muestra matriz de confusión
- Soporta evaluación en conjunto de prueba

## Resultados Obtenidos (hasta ahora):

### Modelo BASE
```
Precisión en test: 83.27%
Pérdida en test: 0.4888
Overfitting: 16.49% (98.22% train - 81.73% val)
Épocas: 8 (early stopping)
Tiempo: 14.14 minutos
```

El modelo base muestra una brecha clara entre entrenamiento y validación, indicativo de overfitting.

## Técnicas Explicadas:

### 1. Data Augmentation
**¿Por qué?** Las redes neuronales profundas necesitan muchos datos. Sin suficientes muestras, aprenden patrones específicos en lugar de características generales.

**¿Cómo?** Aplicamos transformaciones aleatorias a las imágenes de entrenamiento:
- Rotación: el modelo aprende que un "árbol" sigue siendo un árbol aunque esté girado
- Desplazamiento: aprende que los objetos no siempre están centrados
- Zoom: aprende características en diferentes escalas

**Beneficio:** Aumenta efectivamente el tamaño del dataset sin descargar más imágenes.

### 2. Batch Normalization
**¿Por qué?** Los valores activaciones entre capas pueden tener rangos muy diferentes, haciendo el entrenamiento inestable.

**¿Cómo?** Normaliza las salidas de cada capa a media=0, std=1 antes de pasar a la siguiente.

**Beneficio:** Entrenamiento más estable, convergencia más rápida, permite tasas de aprendizaje más altas.

### 3. Dropout
**¿Por qué?** La red puede memorizar patrones ruidosos (overfitting).

**¿Cómo?** Durante el entrenamiento, apagamos aleatoriamente N% de neuronas.

**Beneficio:** Obliga al modelo a ser más robusto y redundante, mejora generalización.

### 4. Learning Rate Scheduling
**¿Por qué?** Una tasa de aprendizaje fija puede ser demasiado grande o pequeña según el progreso.

**¿Cómo?** Si la validación no mejora por N épocas, multiplicamos LR por 0.5.

**Beneficio:** Ajuste fino automático, mejor convergencia en fases finales.

### 5. Early Stopping
**¿Por qué?** Seguir entrenando después de que la validación empeore = overfitting.

**¿Cómo?** Si la pérdida de validación no mejora por 5 épocas, detenemos.

**Beneficio:** Ahorra tiempo, obtiene el mejor modelo automáticamente.

## Tamaño del Dataset

- **Total:** ~25,000 imágenes
- **Entrenamiento:** 14,034 imágenes
- **Prueba:** 3,000 imágenes
- **Clases:** 6 (buildings, forest, glacier, mountain, sea, street)
- **Resolución:** 150x150 píxeles

## Comparación Esperada

| Métrica | BASE | +Augmentation | OPTIMIZADO |
|---------|------|---------------|------------|
| Test Acc | 83.27% | ~85-87% | ~88-90% |
| Overfitting | Alto | Medio | Bajo |
| Tiempo entreno | ~14 min | ~15 min | ~18 min |

## Métricas que se muestran:

1. **Pérdida (Loss)**: Qué tan incorrecto es el modelo
   - CrossEntropyLoss para clasificación multiclase
   - Menor es mejor
   
2. **Precisión (Accuracy)**: % de imágenes clasificadas correctamente
   - 0-1 o 0-100%
   - Mayor es mejor
   
3. **Overfitting**: (Train Acc - Val Acc)
   - Indica si el modelo memoriza vs generaliza
   - Menor es mejor

## Lugar donde aplicar más optimizaciones (opcional):

1. **Transfer Learning**: Usar ResNet/EfficientNet preentrenados en ImageNet
   - Daría ~95%+ de precisión
   - Requiere menos tiempo de entrenamiento
   
2. **Ensemble**: Combinar predicciones de múltiples modelos
   - Mejora 2-3% de precisión
   - Complejidad computacional mayor
   
3. **Hyperparameter Tuning**: Optimizar valores de Dropout, LR, etc.
   - Ganancia pequeña pero consistente
   
4. **Class Weighting**: Si hay desbalance de clases
   - Mejora precisión en clases minoritarias

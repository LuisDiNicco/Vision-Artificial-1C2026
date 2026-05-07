# TP 2 - Clasificación con Machine Learning - Visión Artificial (1C 2026)

## Datos del trabajo práctico
- **Materia:** Visión Artificial
- **Institución:** UNLaM
- **Cuatrimestre:** 1C 2026
- **Entrega:** 7 de mayo

## Integrantes (Grupo 4)

| DNI | Apellido, Nombre |
|---:|---|
| 43.630.151 | Antonioli, Iván Oscar |
| 43.664.669 | Di Nicco, Luis Demetrio |
| 41.069.597 | Rojas, Tomas Ian |

## Consigna
La consigna completa está en: `Consigna TP 2 Clasificacion con machine learning.pdf`

**Resumen:** Construir un sistema que clasifique formas geométricas usando descriptores invariantes (**momentos de Hu**) y un clasificador de Machine Learning entrenado con muestras capturadas en tiempo real desde la webcam.

## Qué hace esta implementación
Sistema completo de clasificación de formas geométricas en tres pasos:

1. **Captura de muestras** (`generadorDescriptores.py`): genera descriptores desde la webcam y guarda en `dataset.csv`
2. **Entrenamiento** (`entrenador.py`): entrena un árbol de decisión con las muestras y genera `modelo.joblib`
3. **Clasificación en tiempo real** (`clasificador.py`): detecta formas y predice su clase

## Formas soportadas
| Código | Forma |
|:---:|---|
| 1 | Cuadrado |
| 2 | Triángulo |
| 3 | Estrella |
| 4 | Hexágono |
| 5 | Círculo |

## Estructura de archivos
- **`generadorDescriptores.py`**: interfaz interactiva para capturar muestras desde webcam
- **`dataset.csv`**: base de datos de muestras con columnas `hu1, hu2, ..., hu7, etiqueta`
- **`entrenador.py`**: entrena el clasificador y genera el modelo
- **`clasificador.py`**: clasifica formas en tiempo real
- **`modelo.joblib`**: archivo con el modelo entrenado (se genera después de ejecutar `entrenador.py`)

## Cómo funciona

### Paso 1: Capturar muestras (`generadorDescriptores.py`)
Ejecutar:
```bash
python generadorDescriptores.py
```

**Interfaz visual:**
- Ventana principal: video en vivo con contorno detectado (coloreado según etiqueta activa)
- Ventana secundaria: máscara binaria (imagen procesada en blanco y negro)
- HUD en pantalla: etiqueta activa, contador de muestras por clase, estado del contorno

**Flujo de captura:**
1. Seleccionar forma con teclas `1` a `5` (se cambia color y nombre en pantalla)
2. Presentar la forma frente a la cámara
3. El sistema detecta automáticamente el contorno más grande
4. Cuando hay contorno válido (área > 90 píxeles), presionar `ESPACIO` para capturar
5. Se calcula el descriptor (7 momentos de Hu) y se guarda en `dataset.csv`
6. Aparece un flash verde para confirmar captura
7. Se muestra conteo actualizado de muestras por clase
8. Repetir con múltiples ángulos, tamaños y posiciones
9. Presionar `Q` para finalizar y cerrar ventanas

**Acciones internas:**
- Conversión BGR → Grises → Desenfoque Gaussiano
- Binarización con threshold inverso (fondo blanco, forma negra)
- Búsqueda de contornos y selección del más grande por área
- Cálculo de 7 momentos de Hu para cada contorno capturado
- Almacenamiento en CSV: `[hu1, hu2, hu3, hu4, hu5, hu6, hu7, etiqueta]` con 10 decimales de precisión

**Resultado:** Archivo `dataset.csv` con todas las muestras capturadas

---

### Paso 2: Entrenar el modelo (`entrenador.py`)
Ejecutar:
```bash
python entrenador.py
```

**Lectura de datos:**
1. Lee `dataset.csv` completo con `pandas`
2. Extrae columnas 0-6 como características (X) = los 7 momentos de Hu
3. Extrae columna 7 como etiquetas (Y) = clase de forma

**Entrenamiento:**
1. Crea un árbol de decisión (`DecisionTreeClassifier` de scikit-learn)
2. Realiza validación cruzada con 5 folds (divide datos en 5 partes, entrena 4 veces)
3. En cada iteración: entrena con 4 folds y valida con el restante
4. Calcula predicciones en todos los datos

**Evaluación y reporte:**
1. Genera **matriz de confusión**: muestra aciertos y errores por clase
2. Genera **classification report**: precisión, recall y F1-score por clase
3. Imprime ambos en consola para análisis de rendimiento

**Exportación:**
- Guarda el modelo entrenado final en `modelo.joblib` usando joblib
- Este archivo contiene todo lo necesario para hacer predicciones

**Salida en consola:**
```
Matriz de confusión:
[[...] [...] ...]
Informe de clasificación:
              precision    recall  f1-score   support
...
```

**Resultado:** Archivo `modelo.joblib` listo para clasificar

---

### Paso 3: Clasificar en tiempo real (`clasificador.py`)
Ejecutar:
```bash
python clasificador.py
```

**Inicialización:**
1. Carga `modelo.joblib` desde disco a memoria
2. Abre la webcam y verifica disponibilidad
3. Imprime en consola que el clasificador está activo

**Ciclo de clasificación (cada frame):**
1. Captura frame de la webcam
2. Espeja la imagen horizontalmente (para que se vea como espejo)
3. Preprocesa: conversión a grises → desenfoque → binarización
4. Busca contornos y filtra por área mínima (500 píxeles)
5. **Para cada contorno válido:**
   - Calcula los 7 momentos de Hu
   - Pasa al modelo para obtener predicción
   - Obtiene nombre de forma según etiqueta predicha
   - **Dibuja en pantalla:**
     - Contorno en color correspondiente a la forma
     - Nombre de forma en la esquina superior izquierda del contorno
6. Dibuja instrucción "Q: salir" en la esquina inferior

**Visualización:**
- La ventana muestra el video en tiempo real
- Cada forma detectada aparece etiquetada con su nombre
- Los colores son consistentes con las 5 clases

**Interacción:**
- Presionar `Q` para terminar y cerrar ventanas

**Resultado:** Clasificación visual en tiempo real de las formas detectadas

## Instalación

Dependencias requeridas:
```bash
pip install opencv-python numpy pandas scikit-learn joblib
```

## Controles

### `generadorDescriptores.py`
| Tecla | Acción |
|---|---|
| `1` - `5` | Cambiar etiqueta activa (cuadrado, triángulo, estrella, hexágono, círculo) |
| `ESPACIO` | Guardar muestra |
| `Q` | Salir |

### `clasificador.py`
| Tecla | Acción |
|---|---|
| `Q` | Salir |

## Detalles técnicos

### Procesamiento de imagen
- **Entrada:** Frame BGR desde webcam
- **Escala de grises:** conversión a espacio gray
- **Desenfoque:** filtro Gaussiano (kernel 5×5) para reducir ruido
- **Binarización:** threshold de **127** con inversión (`THRESH_BINARY_INV`)
  - Resultado: objeto en negro, fondo en blanco
- **Extracción de contornos:** método `RETR_EXTERNAL` busca solo los contornos exteriores

### Parámetros de umbralización
- **UMBRAL_BINARIO:** 127 (rango 0-255)
  - Píxeles por debajo de 127 → negro (objeto)
  - Píxeles por encima de 127 → blanco (fondo)
- **AREA_MIN (generador):** 90 píxeles
  - Detecta contornos relativamente pequeños
  - Permite capturar formas desde distintas distancias
- **AREA_MIN (clasificador):** 500 píxeles
  - Filtra contornos pequeños y ruido
  - Requiere formas más cercanas a la cámara

### Descriptores: Momentos de Hu
Los **7 momentos de Hu** son invariantes matemáticos que describen la forma:
- Invariantes a **rotación, escala y traslación**
- Calculados por OpenCV usando `cv2.HuMoments()` en base a momentos espaciales
- Capturan propiedades geométricas fundamentales de la forma
- Guardados con **precisión de 10 decimales** en CSV para máxima fidelidad

Fórmula: cada momento Hu es una combinación de momentos normalizados que resulta independiente de transformaciones geométricas.

### Clasificador: Árbol de Decisión
- **Algoritmo:** Decision Tree (`DecisionTreeClassifier` de scikit-learn)
- **Entrada:** vector de 7 valores (los momentos de Hu)
- **Salida:** etiqueta de clase (1-5)
- **Validación:** Cross-validation con 5 folds
- **Métricas de evaluación:**
  - **Matriz de confusión:** aciertos en diagonal, errores fuera
  - **Precisión:** porcentaje de predicciones correctas por clase
  - **Recall:** qué porcentaje de cada clase fue detectado
  - **F1-score:** promedio armónico entre precisión y recall
# TP Integrador - Reconocimiento Facial - Visión Artificial (1C 2026)

## Datos del trabajo práctico
- **Materia:** Visión Artificial
- **Institución:** UNLaM
- **Cuatrimestre:** 1C 2026
- **Entrega:** *(a confirmar)*

## Integrantes (Grupo 4)

| DNI | Apellido, Nombre |
|---:|---|
| 43.630.151 | Antonioli, Iván Oscar |
| 43.664.669 | Di Nicco, Luis Demetrio |
| 41.069.597 | Rojas, Tomas Ian |

## Consigna

El proyecto consiste en detectar e identificar (clasificar) rostros en imágenes de webcam, en tiempo real. El output del proyecto es el video anotado en tiempo real detectando las caras y señalando su identidad.

**Requisitos:**
- El sistema debe reconocer al menos a los integrantes del equipo.
- Es fundamental incluir la categoría **"desconocido"** para rostros no registrados.
- Se recomienda usar el siguiente pipeline: detección con MediaPipe, alineamiento con landmarks, extracción de embeddings con ArcFace y clasificación de embeddings para identificar a la persona.

**Pipeline recomendado:**

```
Webcam → Detección (MediaPipe) → Alineamiento (landmarks) → Embeddings (ArcFace) → Clasificación → Identidad
```

## Pipeline: etapas del reconocimiento facial

### 1. Detección de caras (Face Detection)
Localizar una o más caras en la imagen. Es la primera etapa del pipeline.

**Input:** imagen (frame de webcam)
**Output:** un rectángulo delimitador por cada cara detectada, opcionalmente con puntaje de confianza.

**Opciones disponibles:**
| Método | Librería | Observaciones |
|---|---|---|
| MediaPipe Face Detector | `mediapipe` | Rápido, ya visto en TP1, incluye landmarks |
| DNN con Caffe | `opencv-python` | Modelo SSD + ResNet, 300×300 px |
| Haar Cascades (Viola-Jones) | `opencv-python` | Clásico, liviano pero menos robusto |

### 2. Facial Landmarks
Puntos clave sobre el rostro (ojos, nariz, boca, contorno) que permiten:
- Estimar la pose de la cara
- Realizar el alineamiento
- Obtener descriptores geométricos
- Detectar expresiones (boca abierta, ojos cerrados, etc.)

MediaPipe proporciona **478 landmarks faciales**; DLib utiliza la anotación clásica de **68 puntos** del dataset iBug 300-W.

### 3. Alineamiento (Face Alignment)
Consiste en rotar la imagen para que la cara quede vertical, con ambos ojos sobre la misma línea horizontal. El alineamiento mejora significativamente el reconocimiento porque los clasificadores se entrenan con caras verticales y no son invariantes a la rotación.

A partir de los landmarks se obtiene la posición de los ojos, se calcula la transformación y se aplica una rotación que deja:
- Los ojos alineados horizontalmente
- El centro entre ojos centrado horizontalmente
- La imagen final con la resolución requerida por el modelo de embeddings

### 4. Extracción de embeddings
Un **embedding** es un descriptor numérico que representa el rostro como un vector de características.

| Modelo | Dimensiones | Framework | Observaciones |
|---|---|---|---|
| **ArcFace** | 512 | PyTorch / TFLite | Máxima precisión, estado del arte |
| FaceNet | 128 | TensorFlow / Torch | Balance velocidad-precisión |
| DLib | 128 | DLib (C++) | Rápido y ligero |

**ArcFace** es la opción recomendada en el material de clase. Se instala desde PyPI (`arcface`), incluye un modelo entrenado que corre sobre TensorFlow Lite y devuelve embeddings como vectores de **512 elementos de módulo 1**.

Los embeddings se comparan por **distancia euclidiana**: a menor distancia, mayor similitud entre rostros.

### 5. Clasificación / Identificación
A partir de los embeddings, se asigna una identidad a cada rostro. Hay dos enfoques:

**Por distancia (sin entrenamiento):**
- Se calcula un embedding de referencia para cada persona conocida.
- En tiempo real, se compara el embedding del rostro detectado contra todos los de referencia.
- Si la distancia mínima supera un umbral, se clasifica como **"desconocido"**.

**Por machine learning (con entrenamiento):**
- Se entrena un clasificador (SVM, Árbol de Decisión, etc.) usando los embeddings como feature vectors.
- La clase de cada embedding es el nombre de la persona.
- Se agrega una categoría "desconocido" para embeddings que no ajustan a ninguna clase con suficiente confianza.

## Estructura de archivos

```
TP Integrador/
├── README.md                        # Este archivo
├── tp_integrador_gui.py             # Interfaz grafica principal con Dear PyGui
├── src/
│   └── tp_integrador/
│       ├── apps/
│       │   ├── registro.py          # Flujo de registro y entrenamiento
│       │   ├── reconocimiento.py    # Flujo de reconocimiento en tiempo real
│       │   └── gui.py               # Flujo grafico integrado
│       ├── backend/
│       │   ├── deteccion.py         # Detección de rostros con MediaPipe
│       │   ├── alineamiento.py      # Alineamiento facial con landmarks
│       │   ├── embeddings.py        # Extracción de embeddings con ArcFace/DeepFace
│       │   ├── clasificador.py      # Clasificación SVM + desconocido
│       │   ├── camera.py            # Apertura/cambio de webcam
│       │   └── data.py              # Guardado/carga de datos privados
│       └── frontend/
│           └── video_overlay.py     # Anotaciones dibujadas sobre el video
├── datos/
│   └── resumen_embeddings.json      # Resumen no sensible de muestras registradas
├── datos_privados/                  # Embeddings y fotos opcionales (ignorado por Git)
│   ├── embeddings/
│   └── fotos/
├── modelo/
│   └── clasificador_svm.joblib      # Modelo SVM entrenado
└── requirements.txt                 # Dependencias del proyecto
```

## Instalación y ejecución

### Requisitos

Para ejecutar el pipeline completo con ArcFace usar **Python 3.10, 3.11 o 3.12**. Con Python 3.14 pip no puede instalar TensorFlow, dependencia requerida por `arcface`/`deepface`.

En Windows, ejemplo con Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si ya tenías el entorno creado y aparece un error de `astropy` o `tf_keras`, actualizar dependencias:

```powershell
pip install astropy tf-keras deepface
```

Si se instala con Python 3.14, `requirements.txt` omite ArcFace/DeepFace para evitar el conflicto, pero no alcanza para correr la extracción de embeddings del TP.

```bash
pip install -r requirements.txt
```

### Ejecutar

```bash
# Interfaz grafica integrada: registro, entrenamiento y reconocimiento
python tp_integrador_gui.py
```

Por defecto se usa DeepFace con `model_name="ArcFace"`, porque es el camino mas robusto en Python moderno y mantiene el modelo ArcFace del pipeline. Para explicitarlo:

```powershell
$env:TP_FACE_EMBEDDER="deepface"
python tp_integrador_gui.py
```

Para forzar la biblioteca `arcface` directa del material:

```powershell
$env:TP_FACE_EMBEDDER="arcface"
python tp_integrador_gui.py
```

### Controles esperados
| Tecla | Acción |
|---|---|
| `Espacio` | Capturar embedding de la persona actual |
| `N` | Cargar otra persona para registrar |
| `T` | Entrenar y guardar el clasificador SVM |
| `C` | Cambiar a la siguiente cámara disponible |
| `Q` o `ESC` | Salir |

La interfaz muestra una barra superior con el modo activo, un panel inferior con estado/atajos y anotaciones sobre cada rostro con landmarks, caja y etiqueta de identidad/confianza.

### Frontend

La UI principal usa **Dear PyGui**, con panel lateral, selector de camara, botones e inputs sin usar Tkinter. El dibujo de landmarks/cajas sobre el video se hace con OpenCV dentro de `frontend/video_overlay.py`.

En Windows se activa DPI awareness y se carga una fuente del sistema para evitar que la ventana se vea escalada o borrosa en monitores HiDPI.

La ventana es responsive: al agrandarla se reajustan el panel lateral, el video y la escala tipografica. La seleccion de camara no reescanea dispositivos automaticamente; se elige un indice y se confirma con **Cambiar camara**, o se usa **Siguiente** para saltar a la proxima camara que responda.

En modo registro, **Nueva persona** limpia el nombre actual y reinicia el contador de sesion. El panel muestra tambien la cantidad de embeddings guardados para la persona escrita.

Los logs informativos de TensorFlow/TFLite/MediaPipe se silencian al iniciar la app para que la consola no tape los mensajes utiles del TP.

### Doble famoso

El dataset `ares1123/celebrity_dataset` se procesa una sola vez en modo offline. El script descarga el split `train`, alinea cada rostro, calcula embeddings ArcFace por foto y guarda un cache comprimido en `datos_privados/celebrity_dataset/celebrity_arcface_index.npz`.

```powershell
python cache_celebrity_embeddings.py
```

Para una prueba corta:

```powershell
python cache_celebrity_embeddings.py --limit 100 --force
```

El cache guarda dos niveles:
- embeddings individuales por foto, para poder auditar o regenerar;
- un centroide normalizado por famoso, calculado como promedio de sus fotos, con una imagen representativa cercana al centroide.

La app no vuelve a descargar ni reprocesar el dataset durante el reconocimiento. El boton **Buscar doble** toma el embedding actual de la webcam, consulta los centroides cacheados por similitud coseno y muestra el top 5 de famosos distintos en pantalla dividida.

Opciones razonables para una evolucion:
- **Dear PyGui:** liviano y rapido para herramientas visuales con controles, sliders y paneles. Es la opcion incluida en `tp_integrador_gui.py`.
- **PySide6 / Qt:** mejor para una app de escritorio mas completa. Es mas pesado, pero robusto.
- **Web local:** FastAPI + frontend HTML/JS. Se ve mejor, pero integrar webcam + inferencia Python suma complejidad.

## Cómo funciona (resumen)

1. **Registro (offline):** se capturan varias fotos de cada integrante, se extraen sus embeddings con ArcFace y se almacenan como referencia.
2. **En tiempo real:**
   - OpenCV captura el frame de la webcam.
   - MediaPipe Tasks FaceLandmarker detecta los rostros y sus landmarks.
   - Con los landmarks se alinea cada rostro (corrección de rotación).
   - ArcFace extrae el embedding de 512 dimensiones del rostro alineado.
   - El SVM lineal clasifica el embedding y se valida con umbral calibrado por distancia.
   - Si no supera los umbrales, se etiqueta como "desconocido".
   - Se dibuja el rectángulo, los landmarks y el nombre sobre el video.

## Optimizaciones de precisión

- **Alineamiento antes del embedding:** los landmarks de MediaPipe se usan para rotar, escalar y centrar la cara antes de ArcFace/DeepFace.
- **Plantilla ArcFace de 5 puntos:** cuando hay landmarks suficientes, el alineamiento usa ojos, nariz y comisuras de boca con una transformación de similitud hacia la plantilla 112x112 habitual de ArcFace.
- **Filtro de calidad:** no se guardan muestras demasiado chicas, borrosas, oscuras, sobreexpuestas o cortadas por el borde.
- **Comparación contra embeddings reales:** además del centroide por persona, el clasificador conserva los embeddings registrados y compara contra el vecino más cercano.
- **Confianza calibrada:** el porcentaje ya no depende solamente de `predict_proba` del SVM; se calcula con la distancia real respecto de la variación interna de cada persona.
- **Umbral por persona:** al entrenar, se estima un umbral con las distancias leave-one-out de las muestras de cada identidad.
- **Suavizado temporal:** en reconocimiento con un solo rostro, se promedia una ventana corta de predicciones para reducir saltos de confianza entre frames.
- **Doble famoso por centroide:** para evitar que una foto aislada gane por ruido, la busqueda de famosos compara contra el promedio normalizado de todas las fotos disponibles por identidad.

## Referencias

### Material de clase
- Presentación: `Reconocimiento facial.pptx`
- Proyecto: `Proyecto 5_ Reconocimiento de caras.docx`

### ArcFace
- PyPI: https://pypi.org/project/arcface/
- LearnOpenCV: https://learnopencv.com/face-recognition-with-arcface/

### MediaPipe Face Landmarker
- https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

### FaceNet (paper original)
- Schroff et al., 2015: https://arxiv.org/abs/1503.03832

### Conceptos complementarios
- Face detection con DNN (Caffe): `deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel`
- DLib facial landmarks: 68 puntos del dataset iBug 300-W
- Alineamiento facial: rotación para dejar ambos ojos en la misma línea horizontal

## Notas

- La implementación se basa en `Contenidos de Clase/Reconocimiento facial`: detección con landmarks, alineamiento, embeddings y reconocimiento con SVM.
- En Python 3.12 se usa `mediapipe.tasks.vision.FaceLandmarker`. El archivo `face_landmarker.task` se descarga automáticamente en una carpeta temporal corta (`%TEMP%/tp_integrador_mediapipe`) para evitar problemas de rutas largas de Windows.
- ArcFace es el modelo recomendado para embeddings por su precisión superior, usando TensorFlow Lite como backend de inferencia. El extractor soporta `auto`, `arcface` directo y `DeepFace` configurado con `model_name="ArcFace"`.
- La categoría **"desconocido"** es obligatoria: el sistema debe rechazar rostros que no pertenecen a ninguna persona registrada.
- Se sugiere umbral de distancia empírico (ej: 0.6 - 0.8) para decidir entre conocido/desconocido, ajustable según pruebas.

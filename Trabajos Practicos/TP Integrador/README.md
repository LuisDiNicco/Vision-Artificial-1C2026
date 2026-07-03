# TP Integrador - Reconocimiento Facial - Visión Artificial (1C 2026)

## Datos del trabajo práctico
- **Materia:** Visión Artificial
- **Institución:** UNLaM
- **Cuatrimestre:** 1C 2026

## Integrantes (Grupo 4)

| DNI | Apellido, Nombre |
|---:|---|
| 43.630.151 | Antonioli, Iván Oscar |
| 43.664.669 | Di Nicco, Luis Demetrio |
| 41.069.597 | Rojas, Tomas Ian |

## Video de demostración

La ejecución completa del trabajo se presenta en [video_tp_integrador.mp4](assets/entrega/video_tp_integrador.mp4).

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
├── datos_privados/                  # Embeddings propios y fotos opcionales (ignorado por Git)
├── cache/
│   └── famosos/                     # Cache offline del dataset de famosos (ignorado por Git)
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

La fuente de video en la interfaz es solo la **Webcam**. Desde el panel lateral se puede elegir el indice de camara y cambiarlo con **Cambiar camara** o **Siguiente**.

La seccion **Video de famosos** permite elegir un archivo `.mp4`, `.avi`, `.mov`, `.mkv` o `.webm` mediante el explorador nativo del sistema, o pegar una URL de YouTube, y reconocer que actores/famosos aparecen usando el cache `cache/famosos/celebrity_arcface_index.npz`. Para YouTube se usa `yt-dlp`: descarga unicamente la mejor pista de video, sin audio, y la guarda en `cache/videos` antes de analizarla. El reproductor es intencionalmente visual e incluye pausa, reanudacion y una barra para saltar a cualquier instante; estos controles aparecen unicamente en la pestaña **Videos**. **Preprocesar y guardar** analiza primero todo el archivo, muestra porcentaje y tiempo restante, y guarda una linea temporal JSON en `cache/videos/analysis`, incluidos los 478 landmarks completos de cada rostro. Al tocar **Reproducir y reconocer**, la aplicacion busca automaticamente un analisis compatible: si existe lo reutiliza de forma transparente y, si no, ejecuta el reconocimiento en vivo. El cache se invalida automaticamente si cambia el video, los ajustes o el indice de famosos. La reproduccion usa un reloj de tiempo real y descarta con `grab()` los frames atrasados cuando una inferencia demora, evitando reposicionar repetidamente el decodificador y conservando la velocidad original. La vista es responsive y cambia su textura entre 960x540, 1280x720 y 1920x1080 segun el espacio disponible, manteniendo la relacion de aspecto y sin modificar la resolucion original usada por el reconocimiento. Para ahorrar recursos sin perder respuesta visual, MediaPipe actualiza las caras hasta 10 veces por segundo y ArcFace usa por defecto unas 3 inferencias por segundo; las detecciones y predicciones se reutilizan en los frames intermedios. Al igual que la webcam, el reconocimiento de una sola cara promedia los ultimos 5 embeddings y suaviza las ultimas 7 predicciones. Una histeresis conserva brevemente una identidad ante uno o dos frames inciertos, pero reinicia el historial si cambia claramente el rostro, al saltar en el video o luego de una perdida sostenida de deteccion. El analisis descarta rostros chicos, borrosos o poco frontales y compara los embeddings contra los centroides del dataset de famosos. Para videos largos conviene empezar con **Muestreo (seg)** entre `0.33` y `0.45` y subir **Similitud minima** si aparecen falsos positivos.

### Similitud y ajustes del video

ArcFace genera embeddings normalizados y la aplicacion compara el rostro con el centroide de cada famoso mediante **similitud coseno**. El valor mostrado es `similitud * 100`: indica semejanza entre vectores, no una probabilidad de identidad. La distancia coseno equivalente es `1 - similitud`, por lo que maximizar similitud y minimizar distancia producen el mismo orden de candidatos.

- **Intervalo de reconocimiento en vivo:** indica cada cuanto se vuelve a ejecutar ArcFace durante la reproduccion sin cache. Un valor menor actualiza la similitud con mayor frecuencia y consume mas recursos. No modifica el analisis offline, que procesa todos los frames.
- **Similitud minima:** decide si la mejor identidad se acepta o se muestra `desconocido`; no cambia la similitud calculada. Se aplica al modo en vivo y al offline. En vivo se toma al reiniciar la reproduccion. Para cambiar un analisis ya guardado hay que ajustar el valor y volver a usar **Preprocesar y guardar**.
- **Cache en dos niveles:** al reprocesar con otro umbral se reutilizan, cuando son compatibles, las detecciones, landmarks y embeddings costosos. Se recalculan principalmente tracking, aceptacion/rechazo y el JSON final.

En vivo la identidad usa solamente evidencia reciente, porque no conoce frames futuros. El offline consolida cada aparicion mediante un track completo y puede mantener una identidad aunque un frame aislado tenga otro candidato levemente superior. El porcentaje offline varia con la similitud local de cada frame respecto de la identidad global del track; cuando falta un embedding en un hueco corto, se conserva visualmente la ultima similitud medida del mismo track y el JSON la marca como `previous`.

No hay modos de captura de pantalla ni de ventana en la interfaz actual.

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

### Evaluar reconocimiento

El script de evaluacion separa los embeddings propios en entrenamiento/test de forma reproducible y, si existe el cache de famosos, usa embeddings de famosos como negativos para medir la categoria **"desconocido"**.

```powershell
python evaluar_reconocimiento.py
```

Opciones utiles:

```powershell
python evaluar_reconocimiento.py --test-size 0.25 --unknown-limit 300 --seed 42
```

Metricas principales:
- **Accuracy conocidos:** porcentaje de embeddings propios de test clasificados con la identidad correcta.
- **Falsos desconocidos:** porcentaje de embeddings propios rechazados como "desconocido".
- **Rechazo desconocidos:** porcentaje de negativos externos clasificados correctamente como "desconocido".
- **Falsos aceptados:** porcentaje de negativos externos aceptados incorrectamente como una persona registrada.

El reporte completo se guarda en `datos/evaluacion_reconocimiento.json`.

### Controles esperados
| Tecla | Acción |
|---|---|
| `Espacio` | Capturar embedding de la persona actual |
| `N` | Cargar otra persona para registrar |
| `T` | Entrenar y guardar el clasificador SVM |
| `C` | Cambiar a la siguiente cámara disponible |
| `Q` o `ESC` | Salir |

La interfaz muestra una barra superior con el modo activo, un panel inferior con estado/atajos y anotaciones sobre cada rostro con landmarks y caja. En webcam muestra el puntaje del clasificador; tanto el video en vivo como el analisis offline muestran similitud coseno, no una probabilidad. El vivo usa evidencia reciente y el offline asigna la identidad con el track completo.

### Frontend

La UI principal usa **Dear PyGui**, con panel lateral, selector de camara, botones e inputs sin usar Tkinter. El dibujo de landmarks/cajas sobre el video se hace con OpenCV dentro de `frontend/video_overlay.py`.

En Windows se activa DPI awareness y se carga una fuente del sistema para evitar que la ventana se vea escalada o borrosa en monitores HiDPI.

La ventana es responsive: al agrandarla se reajustan el panel lateral, el video y la escala tipografica. La seleccion de camara no reescanea dispositivos automaticamente; se elige un indice y se confirma con **Cambiar camara**, o se usa **Siguiente** para saltar a la proxima camara que responda.

En modo registro, **Nueva persona** limpia el nombre actual y reinicia el contador de sesion. El panel muestra tambien la cantidad de embeddings guardados para la persona escrita.

Los logs informativos de TensorFlow/TFLite/MediaPipe se silencian al iniciar la app para que la consola no tape los mensajes utiles del TP.

### Video de famosos

El dataset `ares1123/celebrity_dataset` se procesa una sola vez en modo offline. El script descarga el split `train`, alinea cada rostro, calcula embeddings ArcFace por foto y guarda un cache comprimido en `cache/famosos/celebrity_arcface_index.npz`.

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

La app no vuelve a descargar ni reprocesar el dataset durante el analisis. Los centroides cacheados se usan para comparar el embedding del rostro del video y mostrar los resultados.

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
- **Alineamiento estable para ArcFace:** con el backend por defecto, MediaPipe localiza landmarks, se aplica una transformacion de 5 puntos compatible con ArcFace y DeepFace extrae el embedding con `detector_backend="skip"`. Si se define `TP_FACE_ALIGNMENT=deepface`, DeepFace vuelve a encargarse del alineamiento interno.
- **Filtro de calidad:** no se guardan muestras demasiado chicas, borrosas, oscuras, sobreexpuestas o cortadas por el borde.
- **Comparación contra embeddings reales:** además del centroide por persona, el clasificador conserva los embeddings registrados y compara contra el vecino más cercano.
- **Confianza calibrada:** el porcentaje ya no depende solamente de `predict_proba` del SVM; se calcula con la distancia real respecto de la variación interna de cada persona.
- **Umbral por persona:** al entrenar, se estima un umbral con las distancias leave-one-out de las muestras de cada identidad.
- **Suavizado temporal:** en reconocimiento con un solo rostro, se promedia una ventana corta de embeddings validos antes de clasificar y luego se suaviza la prediccion para reducir saltos de confianza entre frames.
- **Registro estricto:** para generar datasets mas limpios, las muestras nuevas deben superar un puntaje minimo de calidad antes de guardarse.
- **Filtrado de outliers:** al entrenar, las muestras que quedan estadisticamente lejos del grupo de su propia etiqueta se excluyen del modelo sin borrar los archivos originales.
- **Comparacion por centroide:** para evitar que una foto aislada gane por ruido, la busqueda de famosos compara contra el promedio normalizado de todas las fotos disponibles por identidad.

### Diagnóstico de inestabilidad

En pruebas con webcam puede ocurrir que la prediccion cambie mucho entre frames consecutivos: por ejemplo, pasar de reconocer correctamente a una persona con confianza alta a mostrar **"desconocido"** o confianza cercana a 0. Esto no significa necesariamente que ArcFace falle, sino que el embedding de entrada puede variar por:

- movimiento de la cabeza o pose fuera de frente;
- blur por movimiento o baja velocidad de obturacion;
- iluminacion insuficiente, sombras o sobreexposicion;
- deteccion/crop variable entre frames;
- alineamiento inestable cuando los landmarks de ojos, nariz o boca se mueven;
- expresiones faciales, oclusion parcial, lentes, pelo o mano cerca de la cara;
- uso de un unico frame para decidir identidad.

La bibliografia revisada coincide en que el reconocimiento facial no debe interpretarse como una etapa aislada, sino como un pipeline completo: deteccion robusta, landmarks confiables, alineamiento, extraccion de embeddings y clasificacion con umbrales adecuados.

## Referencias

### Material de clase
- Presentación: `Reconocimiento facial.pptx`
- Proyecto: `Proyecto 5_ Reconocimiento de caras.docx`

### ArcFace
- ArcFace en PyPI: https://pypi.org/project/arcface/
- Implementacion TensorFlow 2 usada como base por la libreria: https://github.com/peteryuX/arcface-tf2
- Guia conceptual y practica de ArcFace: https://learnopencv.com/face-recognition-with-arcface/
- Paper original, ArcFace: Additive Angular Margin Loss for Deep Face Recognition: https://arxiv.org/abs/1801.07698

### MediaPipe Face Landmarker
- https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

### FaceNet (paper original)
- Schroff et al., 2015: https://arxiv.org/abs/1503.03832

### Deteccion, landmarks, alineamiento y clasificacion
- Face detection con OpenCV DNN/Caffe: https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
- Facial landmarks con DLib y OpenCV: https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
- Alineamiento facial con OpenCV: https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
- Reconocimiento facial con embeddings y SVM: https://pyimagesearch.com/2018/09/24/opencv-face-recognition/
- Clustering facial para agrupar embeddings y detectar outliers: https://pyimagesearch.com/2018/07/09/face-clustering-with-python/

## Notas

- La implementación se basa en `Contenidos de Clase/Reconocimiento facial`: detección con landmarks, alineamiento, embeddings y reconocimiento con SVM.
- En Python 3.12 se usa `mediapipe.tasks.vision.FaceLandmarker`. El archivo `face_landmarker.task` se descarga automáticamente en una carpeta temporal corta (`%TEMP%/tp_integrador_mediapipe`) para evitar problemas de rutas largas de Windows.
- ArcFace es el modelo recomendado para embeddings por su precisión superior, usando TensorFlow Lite como backend de inferencia. El extractor soporta `auto`, `arcface` directo y `DeepFace` configurado con `model_name="ArcFace"`.
- La categoría **"desconocido"** es obligatoria: el sistema debe rechazar rostros que no pertenecen a ninguna persona registrada.

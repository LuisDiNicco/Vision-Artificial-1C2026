# TP 1 - Vision Artificial (1C 2026)

## Datos del trabajo practico
- Materia: Vision Artificial
- Institucion: UNLaM
- Cuatrimestre: 1C 2026
- Entrega: 23 de abril

## Integrantes (Grupo 4)

| DNI | Apellido, Nombre |
|---:|---|
| 43.630.151 | Antonioli, Iván Oscar |
| 43.664.669 | Di Nicco, Luis Demetrio |
| 41.069.597 | Rojas, Tomas Ian |
## Consigna
Instalar MediaPipe y aplicar alguna solucion de vision artificial.

Guia oficial de referencia:
- https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419

## Implementacion actual (Grupo 4)
La version actual en [tp1_main_grupo_4.py](tp1_main_grupo_4.py) implementa una calculadora por mano en tiempo real usando **MediaPipe Hand Landmarker**.

Que hace:
- Detecta landmarks de una mano con webcam (modo `VIDEO` de MediaPipe Tasks).
- Interpreta digitos del 0 al 5 segun dedos extendidos.
- Permite armar operaciones matematicas simples y calcular resultado.
- Muestra una interfaz separada de la camara (panel a la derecha), para no tapar el video.

## Estructura de archivos (modularizada)
Se separo el codigo en archivos mas chicos para mantenerlo entendible:

- [tp1_main_grupo_4.py](tp1_main_grupo_4.py): flujo principal (captura de camara, ciclo de frames y control de estados).
- [tp1_vision.py](tp1_vision.py): todo lo relacionado a vision artificial.
  - descarga de modelo `.task`
  - inicializacion de `HandLandmarker`
  - deteccion de landmarks y mapeo de mano a digitos
  - dibujo de landmarks y conexiones
- [tp1_helpers_ui.py](tp1_helpers_ui.py): helpers y capa de interfaz.
  - estado de calculadora
  - estabilizacion temporal de detecciones
  - operaciones matematicas
  - render del panel lateral y textos

## Referencias oficiales usadas
- Guia general de MediaPipe Solutions:
  - https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419
- Hand Landmarker (documentacion):
  - https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=es-419
- Hand Landmarker (ejemplo Python/Colab):
  - https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb?hl=es-419

## Como funciona (resumen)
1. Se inicializa el Task de mano con `RunningMode.VIDEO`.
2. En cada frame:
  - OpenCV captura la imagen en BGR.
  - Se convierte a RGB y se envuelve en `mp.Image`.
  - Se ejecuta `detect_for_video(...)` con timestamp en milisegundos.
3. Con los landmarks detectados, se estima que dedos estan extendidos.
4. Ese patron de dedos se mapea a un digito de 0 a 5.
5. Un estabilizador temporal evita que el numero cambie por ruido entre frames.
6. El usuario confirma el numero y arma la operacion por teclado.

## Controles
- `Enter`: confirmar numero detectado
- `+ - * /`: seleccionar operador
- `r`: reset
- `q`: salir

## Instalacion y ejecucion
1. Instalar dependencias:

```bash
pip install mediapipe opencv-python numpy
```

2. Ejecutar desde esta carpeta:

```bash
python "tp1_main_grupo_4.py"
```

## Notas de funcionamiento
- El modelo `hand_landmarker.task` se descarga automaticamente si no existe.
- Se aplican validaciones geometricas del pulgar para mejorar la diferencia entre 4 y 5.
- La deteccion de gestos fue removida para dejar el caso de uso enfocado solo en numeros con mano.

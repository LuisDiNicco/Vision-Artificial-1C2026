# TP 1 - Operaciones matematicas con MediaPipe

## Que solucion de MediaPipe implementa
Este trabajo implementa una calculadora por mano en tiempo real usando MediaPipe Tasks (Vision):

- Hand Landmarker: para detectar landmarks de la mano y estimar digitos de 0 a 5.

La app usa webcam y OpenCV para procesar video frame a frame.

## Referencias oficiales usadas
- Guia general de soluciones MediaPipe:
  - https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419
- Hand Landmarker (documentacion):
  - https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=es-419
- Hand Landmarker (ejemplo Python/Colab):
  - https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb?hl=es-419

## Como funciona (resumen)
1. Inicializa el Task de mano en modo VIDEO:
   - `HandLandmarker.create_from_options(...)`
2. En cada frame:
   - Convierte BGR a RGB.
   - Crea `mp.Image`.
   - Ejecuta deteccion por timestamp con `detect_for_video`.
3. Usa landmarks para deducir que dedos estan extendidos y mapear un digito (0-5).
4. Aplica estabilizacion temporal para reducir ruido y falsos positivos.
5. Muestra overlay con estado, operacion, resultado y advertencias de luz.

## Controles
- Teclado (se mantiene):
  - `Enter`: confirmar numero
  - `+ - * /`: operador
  - `r`: reset
  - `q`: salir

## Nota breve de implementacion
El codigo fue ajustado para seguir la forma de trabajo recomendada por Google en los Colab oficiales: configuracion por `BaseOptions` + `...Options`, uso de `RunningMode.VIDEO`, entrada `mp.Image` y deteccion de landmarks de mano en tiempo real.

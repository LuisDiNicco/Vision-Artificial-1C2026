# TP 1 - Vision Artificial (1C 2026)

## Datos del trabajo practico

- Materia: Vision Artificial
- Carrera/Institucion: UNLaM
- Cuatrimestre: 1C 2026
- Entrega: 23 de abril

## Consigna

Instalar MediaPipe y aplicar alguna de las soluciones relacionadas a vision artificial.

Ejemplos propuestos en la consigna:
- Clasificador de gatos
- Segmentador para eliminar el fondo de una imagen y resaltar un objeto o persona

Guia oficial de referencia:
- https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419

## Archivo principal: TP 1 Grupo 4

El archivo `TP 1 Grupo 4.py` implementa deteccion de objetos en tiempo real con webcam usando MediaPipe Tasks (Vision).

Resumen de funcionamiento:
- Usa OpenCV para capturar video desde la camara
- Usa MediaPipe Object Detector en modo `LIVE_STREAM`
- Dibuja bounding boxes y etiqueta cada objeto detectado con su confianza
- Muestra panel con modelo activo, FPS y cantidad de detecciones
- Permite alternar entre modelos `EfficientDet-Lite0` y `EfficientDet-Lite2`

## Librerias utilizadas

- mediapipe
- opencv-python

## Instalacion y ejecucion

1. Instalar dependencias:

```bash
pip install mediapipe opencv-python
```

2. Ejecutar:

```bash
python "TP 1 Grupo 4.py"
```

## Teclas utiles

- `q`: salir
- `0`: cambiar a modelo EfficientDet-Lite0
- `2`: cambiar a modelo EfficientDet-Lite2

Nota: los modelos `.tflite` se descargan automaticamente la primera vez que se usan.
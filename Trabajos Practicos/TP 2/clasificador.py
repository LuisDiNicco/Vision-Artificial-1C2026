"""
Clasificador - Visión Artificial
=================================
Carga el modelo entrenado y clasifica formas en tiempo real
usando la webcam.

Uso:
  python clasificador.py

Controles:
  Q → salir
"""

import cv2
import numpy as np
from joblib import load

# ── Configuración ─────────────────────────────────────────────────────────────
MODELO_PATH    = "modelo.joblib"
UMBRAL_BINARIO = 127
AREA_MIN       = 500

ETIQUETAS = {
    1: "cuadrado",
    2: "triangulo",
    3: "estrella",
}

COLORES = {
    1: (0, 200, 255),
    2: (255, 100, 0),
    3: (0, 255, 128),
}
COLOR_DEFAULT = (200, 200, 200)

# ── Carga del modelo ──────────────────────────────────────────────────────────
print("Cargando modelo...")
clasificador = load(MODELO_PATH)
print("Modelo cargado.")

# ── Funciones ─────────────────────────────────────────────────────────────────
def preprocesar(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(blur, UMBRAL_BINARIO, 255, cv2.THRESH_BINARY_INV)
    return binaria

def obtener_contornos(binaria):
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contornos if cv2.contourArea(c) >= AREA_MIN]

def calcular_hu(contorno):
    momentos = cv2.moments(contorno)
    hu = cv2.HuMoments(momentos).flatten()
    return hu

# ── Main ──────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: No se pudo abrir la webcam.")
    exit()

print("Clasificador activo. Presioná Q para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    binaria = preprocesar(frame)
    contornos = obtener_contornos(binaria)

    for contorno in contornos:
        hu = calcular_hu(contorno).reshape(1, -1)
        etiqueta = clasificador.predict(hu)[0]

        nombre = ETIQUETAS.get(etiqueta, f"clase {etiqueta}")
        color  = COLORES.get(etiqueta, COLOR_DEFAULT)

        # Dibujar contorno y etiqueta
        cv2.drawContours(frame, [contorno], -1, color, 2)
        x, y, _, _ = cv2.boundingRect(contorno)
        cv2.putText(frame, nombre, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, "Q: salir", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Clasificador", frame)

    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
"""
Generador de Descriptores - Visión Artificial
=============================================
Uso:
  1. Presentá la forma frente a la webcam sobre un fondo claro.
  2. Usá las teclas 1, 2, 3... para seleccionar la etiqueta activa.
  3. Presioná ESPACIO para guardar los invariantes de Hu en el CSV.
  4. Presioná Q para salir.

El CSV resultante (dataset.csv) tiene 8 columnas:
  hu1, hu2, hu3, hu4, hu5, hu6, hu7, etiqueta
"""

import cv2
import numpy as np
import csv
import os

# ── Diccionario de etiquetas ─────────────────────────────────────────────────
ETIQUETAS = {
    1: "cuadrado",
    2: "triangulo",
    3: "estrella",
}

# ── Configuración ─────────────────────────────────────────────────────────────
CSV_PATH       = "dataset.csv"
UMBRAL_BINARIO = 127   # umbral para binarización (ajustá según iluminación)
AREA_MIN       = 500   # área mínima del contorno para considerarlo válido

# Colores por etiqueta (BGR) para feedback visual
COLORES = {
    1: (0, 200, 255),   # amarillo-naranja → cuadrado
    2: (255, 100, 0),   # azul             → triángulo
    3: (0, 255, 128),   # verde            → estrella
}
COLOR_DEFAULT = (200, 200, 200)

def preprocesar(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(blur, UMBRAL_BINARIO, 255, cv2.THRESH_BINARY_INV)
    return binaria

def obtener_contorno_principal(binaria):
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    contorno = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(contorno) < AREA_MIN:
        return None
    return contorno

def calcular_hu(contorno):
    momentos = cv2.moments(contorno)
    hu = cv2.HuMoments(momentos).flatten()
    return hu

def guardar_en_csv(hu, etiqueta, path):
    """Agrega una fila al CSV. Crea el archivo con encabezado si no existe."""
    archivo_nuevo = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if archivo_nuevo:
            writer.writerow(["hu1","hu2","hu3","hu4","hu5","hu6","hu7","etiqueta"])
        writer.writerow([f"{v:.10e}" for v in hu] + [etiqueta])

def contar_muestras_por_etiqueta(path):
    """Lee el CSV y devuelve un dict {etiqueta: cantidad}."""
    conteo = {}
    if not os.path.exists(path):
        return conteo
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = int(row["etiqueta"])
            conteo[e] = conteo.get(e, 0) + 1
    return conteo

def dibujar_hud(display, etiqueta_activa, conteo, contorno_ok, flash):
    h, w = display.shape[:2]
    color_activo = COLORES.get(etiqueta_activa, COLOR_DEFAULT)
    nombre_activo = ETIQUETAS.get(etiqueta_activa, f"clase {etiqueta_activa}")

    # ── Panel superior ────────────────────────────────────────────────────────
    cv2.rectangle(display, (0, 0), (w, 70), (30, 30, 30), -1)

    # Etiqueta activa
    cv2.putText(display, f"Etiqueta activa:", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(display, f"[{etiqueta_activa}] {nombre_activo.upper()}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_activo, 2)

    # Conteo de muestras (derecha)
    linea_y = 18
    cv2.putText(display, "Muestras guardadas:", (w - 210, linea_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    for eti, nombre in ETIQUETAS.items():
        linea_y += 18
        cant = conteo.get(eti, 0)
        color_c = COLORES.get(eti, COLOR_DEFAULT)
        cv2.putText(display, f"[{eti}] {nombre}: {cant}",
                    (w - 210, linea_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color_c, 1)

    # ── Estado del contorno ───────────────────────────────────────────────────
    if contorno_ok:
        msg  = "Contorno OK  -  ESPACIO para guardar"
        col  = color_activo
    else:
        msg  = "Sin contorno valido"
        col  = (0, 0, 220)
    cv2.putText(display, msg, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # ── Flash verde al guardar ────────────────────────────────────────────────
    if flash > 0:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la webcam.")
        return

    print("=" * 55)
    print("  GENERADOR DE DESCRIPTORES  |  Visión Artificial")
    print("=" * 55)
    print(f"  CSV de salida: {os.path.abspath(CSV_PATH)}")
    print()
    for num, nombre in ETIQUETAS.items():
        print(f"  Tecla {num} → {nombre}")
    print("  ESPACIO → guardar muestra")
    print("  Q       → salir")
    print("=" * 55)

    etiqueta_activa = 1
    conteo = contar_muestras_por_etiqueta(CSV_PATH)
    flash  = 0   # frames restantes del flash visual

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        binaria = preprocesar(frame)
        contorno = obtener_contorno_principal(binaria)
        display  = frame.copy()

        color_activo = COLORES.get(etiqueta_activa, COLOR_DEFAULT)

        # Dibujar contorno detectado
        if contorno is not None:
            cv2.drawContours(display, [contorno], -1, color_activo, 2)

        dibujar_hud(display, etiqueta_activa, conteo,
                    contorno is not None, flash)

        if flash > 0:
            flash -= 1

        cv2.imshow("Generador de Descriptores", display)
        cv2.imshow("Mascara binaria", binaria)

        tecla = cv2.waitKey(1) & 0xFF

        # Salir
        if tecla in (ord('q'), ord('Q')):
            print(f"\nDataset guardado en: {os.path.abspath(CSV_PATH)}")
            print(f"Total de muestras: {sum(conteo.values())}")
            break

        # Cambiar etiqueta con teclas numéricas
        elif chr(tecla).isdigit() and int(chr(tecla)) in ETIQUETAS:
            etiqueta_activa = int(chr(tecla))
            nombre = ETIQUETAS[etiqueta_activa]
            print(f"Etiqueta cambiada a [{etiqueta_activa}] {nombre}")

        # Capturar muestra
        elif tecla == ord(' '):
            if contorno is not None:
                hu = calcular_hu(contorno)
                guardar_en_csv(hu, etiqueta_activa, CSV_PATH)
                conteo[etiqueta_activa] = conteo.get(etiqueta_activa, 0) + 1
                flash = 6  # duración del flash (frames)
                nombre = ETIQUETAS[etiqueta_activa]
                total = sum(conteo.values())
                print(f"  Guardada: [{etiqueta_activa}] {nombre} "
                      f"(total {nombre}: {conteo[etiqueta_activa]}, "
                      f"total general: {total})")
            else:
                print("  ⚠  Sin contorno válido para capturar.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
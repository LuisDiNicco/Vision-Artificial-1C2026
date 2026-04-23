import cv2, math
import numpy as np
import mediapipe as mp

# =========================
# Helpers geométricos
# =========================
def v2(p): return np.array([p.x, p.y])
def v3(p): return np.array([p.x, p.y, p.z])

def angle(a, b, c):
    """Ángulo ABC en grados (entre BA y BC). Usa 2D (x,y)."""
    ba = v2(a) - v2(b)
    bc = v2(c) - v2(b)
    den = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = np.clip(np.dot(ba, bc) / den, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def dist2(a, b):  # distancia 2D normalizada (x,y)
    return np.linalg.norm(v2(a) - v2(b))

def dist3(a, b):  # distancia 3D normalizada (x,y,z)
    return np.linalg.norm(v3(a) - v3(b))

def palm_center(lm):
    """Centro aproximado de la palma: promedio de 0,5,9,13,17."""
    idxs = [0, 5, 9, 13, 17]
    pts = np.array([[lm[i].x, lm[i].y] for i in idxs], dtype=float)
    c = pts.mean(axis=0)
    class P: pass
    p = P(); p.x, p.y = c[0], c[1]
    return p

# =========================
# Reglas de dedo extendido
# =========================
# Índices MediaPipe: https://google.github.io/mediapipe/solutions/hands#hand-landmark-model
# Para dedos (excepto pulgar) usamos el ángulo en PIP (punto intermedio):
#   si es muy recto (> 160°) lo consideramos extendido.
FINGERS = {
    "index":  (5, 6, 8),   # (MCP, PIP, TIP)
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}
# Para pulgar usamos dos criterios: ángulo en MCP y orientación lateral segun "handedness"
THUMB = (2, 3, 4)  # (CMC, MCP, TIP)

def finger_extended(lm, mcp, pip, tip, ang_thr=160):
    return angle(lm[mcp], lm[pip], lm[tip]) > ang_thr

def thumb_extended(lm, handedness, ang_thr=160):
    cmc, mcp, tip = THUMB
    ang = angle(lm[cmc], lm[mcp], lm[tip])
    # Orientación lateral (eje X) ayuda a distinguir mano izq/der
    # Right: tip.x < mcp.x cuando está extendido hacia la derecha del usuario (imagen espejada)
    # Left : tip.x > mcp.x
    lateral_ok = (lm[tip].x < lm[mcp].x) if handedness == "Right" else (lm[tip].x > lm[mcp].x)
    return (ang > ang_thr) and lateral_ok

def thumb_folded_hint(lm):
    """Heurística extra para pulgar doblado: tip cerca del centro de la palma y 'más profundo' (z)."""
    c = palm_center(lm)
    width = dist2(lm[5], lm[17]) + 1e-6
    close_to_palm = dist2(lm[4], c) / width < 0.6
    # Si tip.z es mayor (menos negativo) que mcp.z, suele estar 'por detrás' (doblado hacia la palma)
    z_deeper = (lm[4].z - lm[3].z) > 0.02
    return close_to_palm or z_deeper

# =========================
# Clasificación de gesto (RPSLS)
# =========================
def classify_hand(lm, handedness):
    """
    Devuelve uno de: 'Piedra','Papel','Tijera','Lagarto','Spock','NADA'
    """
    # Estado de dedos
    ext = {}
    for name, (mcp, pip, tip) in FINGERS.items():
        ext[name] = finger_extended(lm, mcp, pip, tip, ang_thr=160)

    thumb_ext = thumb_extended(lm, handedness, ang_thr=160)
    # Conteo total
    count = int(thumb_ext) + sum(ext.values())

    # Distancias útiles
    width = dist2(lm[5], lm[17]) + 1e-6
    # separaciones entre puntas para detectar 'Spock'
    d_mid_ring = dist3(lm[12], lm[16])
    d_idx_pinky = dist3(lm[8], lm[20])
    # pinza pulgar-índice normalizada (para Lagarto)
    pinch_norm = dist2(lm[4], lm[8]) / width

    # Reglas base
    if count == 0:
        return "Piedra"
    if count == 2 and ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"] and not thumb_ext:
        return "Tijera"
    if count == 5:
        # Spock vs Papel: en Spock hay 'V' entre medio/anular
        if d_mid_ring > 0.55 * d_idx_pinky:
            return "Spock"
        else:
            return "Papel"
    if count == 4:
        # Candidato a Lagarto: cuatro extendidos y pulgar doblado
        # Confirmamos pulgar doblado con heurística extra y pinza moderada
        if not thumb_ext and thumb_folded_hint(lm) and pinch_norm < 0.45:
            return "Lagarto"
        # si no cumple condiciones, probablemente es Papel mal contado
        return "Papel"

    # Casos intermedios/ambiguos: intentar decidir
    # - posible 'Lagarto' con pinky dudoso: 3 extendidos (index, middle, ring), pulgar doblado y pinza chica
    if (ext["index"] and ext["middle"] and ext["ring"] and not ext["pinky"] and not thumb_ext
        and pinch_norm < 0.4 and thumb_folded_hint(lm)):
        return "Lagarto"

    # - posible 'Papel' incompleto (4 o 5 dedos extendidos)
    if sum([ext["index"], ext["middle"], ext["ring"], ext["pinky"]]) >= 3 and (thumb_ext or not thumb_folded_hint(lm)):
        # Diferenciar Spock si hay gran separación medio-anular
        if d_mid_ring > 0.6 * d_idx_pinky:
            return "Spock"
        return "Papel"

    return "NADA"

# =========================
# Reglas de victoria RPSLS
# =========================
BEATS = {
    "Piedra": ["Tijera", "Lagarto"],
    "Papel":  ["Piedra", "Spock"],
    "Tijera": ["Papel", "Lagarto"],
    "Lagarto":["Papel", "Spock"],
    "Spock":  ["Piedra", "Tijera"],
}

def decide_winner(gesto_L, gesto_R):
    if gesto_L == "NADA" or gesto_R == "NADA":
        return "Esperando gestos validos…"
    if gesto_L == gesto_R:
        return f"Empate ({gesto_L})"
    if gesto_R in BEATS.get(gesto_L, []):
        return f"Gana IZQUIERDA ({gesto_L} vence a {gesto_R})"
    if gesto_L in BEATS.get(gesto_R, []):
        return f"Gana DERECHA ({gesto_R} vence a {gesto_L})"
    return "Sin decisión"

# =========================
# Main: detección 2 manos
# =========================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_styles= mp.solutions.drawing_styles

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # probá CAP_MSMF si DSHOW no abre en tu entorno
    with mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)

            # Mapear por handedness
            gestos = {"Left":"NADA", "Right":"NADA"}

            if res.multi_hand_landmarks:
                for i, hand in enumerate(res.multi_hand_landmarks):
                    # dibujar landmarks
                    mp_draw.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                    # handedness (Left/Right)
                    label = "Right"
                    if res.multi_handedness and len(res.multi_handedness) > i:
                        label = res.multi_handedness[i].classification[0].label
                    # clasificar gesto
                    g = classify_hand(hand.landmark, label)
                    gestos[label] = g
                    # pintar etiqueta sobre cada mano (usar la muñeca como ancla)
                    wrist = hand.landmark[0]
                    h, w = frame.shape[:2]
                    x, y = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(frame, f"{label[:1]}: {g}", (x+10, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50), 2, cv2.LINE_AA)

            # decidir ganador IZQ vs DER
            resultado = decide_winner(gestos["Left"], gestos["Right"])
            cv2.rectangle(frame, (10,10), (10+640, 10+40), (0,0,0), -1)
            cv2.putText(frame, resultado, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)

            cv2.imshow("RPSLS - 2 manos", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
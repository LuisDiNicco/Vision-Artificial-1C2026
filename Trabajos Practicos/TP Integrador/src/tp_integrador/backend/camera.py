import cv2


def open_webcam(index: int = 0) -> cv2.VideoCapture:
    """Abre la webcam intentando primero el backend DirectShow en Windows."""
    backends = [getattr(cv2, "CAP_DSHOW", 700), cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def list_available_cameras(max_index: int = 8) -> list[int]:
    """Devuelve indices de camara que OpenCV puede abrir."""
    available = []
    for index in range(max_index + 1):
        cap = open_webcam(index)
        ok = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            if ret:
                available.append(index)
        cap.release()
    return available


def next_available_camera(current_index: int, max_index: int = 8) -> tuple[int, cv2.VideoCapture] | tuple[None, None]:
    """Abre la siguiente camara disponible despues de current_index."""
    candidates = list(range(current_index + 1, max_index + 1)) + list(range(0, current_index + 1))
    for index in candidates:
        if index == current_index:
            continue
        cap = open_webcam(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return index, cap
        cap.release()
    return None, None

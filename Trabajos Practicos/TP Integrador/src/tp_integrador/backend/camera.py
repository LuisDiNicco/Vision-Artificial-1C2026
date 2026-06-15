import cv2


def open_webcam(
    index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> cv2.VideoCapture:
    """Abre la webcam intentando primero el backend DirectShow en Windows."""
    backends = [getattr(cv2, "CAP_DSHOW", 700), cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            configure_capture(cap, width, height, fps)
            return cap
        cap.release()
    cap = cv2.VideoCapture(index)
    configure_capture(cap, width, height, fps)
    return cap


def configure_capture(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> None:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


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

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

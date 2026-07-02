import logging
import os
import threading


_NATIVE_FILTER_INSTALLED = False
_NOISY_NATIVE_MESSAGES = (
    b"face_landmarker_graph.cc",
    b"inference_feedback_manager.cc",
    b"Created TensorFlow Lite XNNPACK delegate",
    b"portable_clearcut_uploader.cc",
    b"Failed to send to clearcut",
)


def _install_native_stderr_filter() -> None:
    """Filtra ruido conocido de librerias C++ conservando el resto de stderr."""
    global _NATIVE_FILTER_INSTALLED
    if _NATIVE_FILTER_INSTALLED or not hasattr(os, "pipe"):
        return
    try:
        original_stderr = os.dup(2)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)
        os.close(write_fd)
    except OSError:
        return

    _NATIVE_FILTER_INSTALLED = True

    def forward_stderr() -> None:
        suppress_trace_lines = 0
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            for line in iter(stream.readline, b""):
                if any(message in line for message in _NOISY_NATIVE_MESSAGES):
                    suppress_trace_lines = 3 if b"clearcut" in line or b"portable_clearcut" in line else 0
                    continue
                if suppress_trace_lines and (
                    b"Source Location Trace" in line
                    or b"wireless/android/play/playlog/" in line
                    or not line.strip()
                ):
                    suppress_trace_lines -= 1
                    continue
                suppress_trace_lines = 0
                try:
                    os.write(original_stderr, line)
                except OSError:
                    break

    threading.Thread(target=forward_stderr, name="native-stderr-filter", daemon=True).start()


def configure_native_logs() -> None:
    """Reduce ruido de TensorFlow, TFLite y MediaPipe.

    Debe ejecutarse antes de importar modulos que carguen TensorFlow/MediaPipe.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["GLOG_stderrthreshold"] = "3"
    os.environ["ABSL_LOGGING_MIN_LOG_LEVEL"] = "3"
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    _install_native_stderr_filter()
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("mediapipe").setLevel(logging.ERROR)


def configure_absl_logs() -> None:
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
        absl_logging.set_stderrthreshold("fatal")
    except Exception:
        pass

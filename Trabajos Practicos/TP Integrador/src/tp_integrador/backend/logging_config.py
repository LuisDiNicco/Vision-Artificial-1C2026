import logging
import os


def configure_native_logs() -> None:
    """Reduce ruido de TensorFlow, TFLite y MediaPipe.

    Debe ejecutarse antes de importar modulos que carguen TensorFlow/MediaPipe.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_logtostderr", "0")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("mediapipe").setLevel(logging.ERROR)


def configure_absl_logs() -> None:
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass

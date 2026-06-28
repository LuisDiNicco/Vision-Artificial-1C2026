from pathlib import Path
import faulthandler
import sys


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

CRASH_LOG_DIR = ROOT_DIR / "cache" / "logs"
CRASH_LOG_DIR.mkdir(parents=True, exist_ok=True)
CRASH_LOG_STREAM = (CRASH_LOG_DIR / "native_crash.log").open("a", encoding="utf-8")
faulthandler.enable(file=CRASH_LOG_STREAM, all_threads=True)

from tp_integrador.backend.logging_config import configure_native_logs


configure_native_logs()

from tp_integrador.apps.gui import main


if __name__ == "__main__":
    main()

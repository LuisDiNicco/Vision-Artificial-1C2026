from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from tp_integrador.backend.logging_config import configure_native_logs


configure_native_logs()

from tp_integrador.apps.gui import main


if __name__ == "__main__":
    main()

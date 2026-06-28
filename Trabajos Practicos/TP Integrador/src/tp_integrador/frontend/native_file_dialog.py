from pathlib import Path
from typing import Optional


VIDEO_FILE_TYPES = (
    ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
    ("Todos los archivos", "*.*"),
)


def choose_video_file(initial_path: Optional[str] = None) -> Optional[Path]:
    """Abre el selector nativo del sistema y devuelve el video elegido."""
    import tkinter as tk
    from tkinter import filedialog

    initial_dir = _initial_directory(initial_path)
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update_idletasks()
    try:
        selected = filedialog.askopenfilename(
            parent=root,
            title="Elegir video local",
            initialdir=str(initial_dir),
            filetypes=VIDEO_FILE_TYPES,
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None


def _initial_directory(initial_path: Optional[str]) -> Path:
    if initial_path:
        path = Path(initial_path)
        if path.is_file():
            return path.parent
        if path.is_dir():
            return path
    return Path.home()

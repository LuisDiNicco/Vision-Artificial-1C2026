from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from .data import BASE_DIR


VIDEO_DOWNLOAD_DIR = BASE_DIR / "cache" / "videos"


def looks_like_youtube_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.netloc.lower()
    return host.endswith("youtube.com") or host.endswith("youtu.be")


def download_youtube_video(url: str, progress_callback=None) -> Path:
    """Descarga un video de YouTube a cache local y devuelve su ruta.

    Usa yt-dlp si esta instalado. El analisis posterior trabaja sobre archivo
    local, igual que con videos seleccionados desde disco.
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("Falta yt-dlp. Instala dependencias con: pip install -r requirements.txt") from exc

    VIDEO_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def hook(status) -> None:
        if progress_callback is None:
            return
        if status.get("status") == "downloading":
            percent = status.get("_percent_str", "").strip()
            speed = status.get("_speed_str", "").strip()
            progress_callback(f"Descargando YouTube... {percent} {speed}".strip())
        elif status.get("status") == "finished":
            progress_callback("Video descargado. Preparando analisis...")

    options = {
        # Solo se usa imagen: descarga directamente la mejor pista de video.
        "format": "bestvideo/best[vcodec!=none]",
        "outtmpl": str(VIDEO_DOWNLOAD_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        path = _find_downloaded_video(info, ydl)
        if path is None:
            raise RuntimeError("yt-dlp termino, pero no se encontro el archivo descargado.")
        return path


def _find_downloaded_video(info: dict, ydl) -> Path | None:
    candidates = []
    for key in ("filepath", "_filename"):
        if info.get(key):
            candidates.append(Path(info[key]))
    candidates.append(Path(ydl.prepare_filename(info)))

    video_id = str(info.get("id", "")).strip()
    if video_id:
        candidates.extend(
            sorted(
                VIDEO_DOWNLOAD_DIR.glob(f"{video_id}.*"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
        )

    video_suffixes = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in video_suffixes:
            return candidate
    return None

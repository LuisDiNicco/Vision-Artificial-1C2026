from pathlib import Path
import sys
import argparse


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from tp_integrador.backend.logging_config import configure_native_logs


configure_native_logs()

from tp_integrador.backend.celebrity import CELEBRITY_INDEX_PATH, load_or_build_celebrity_cache
from tp_integrador.backend.embeddings import ArcFaceEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera cache de embeddings ArcFace para famosos.")
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo las primeras N filas del dataset.")
    parser.add_argument("--max-per-person", type=int, default=None, help="Maximo de imagenes por famoso.")
    parser.add_argument("--force", action="store_true", help="Regenerar aunque ya exista el cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedder = ArcFaceEmbedder()
    index = load_or_build_celebrity_cache(
        embedder,
        force=args.force,
        limit=args.limit,
        max_per_person=args.max_per_person,
    )
    print(f"Cache listo: {len(index.names)} famosos en {CELEBRITY_INDEX_PATH}")


if __name__ == "__main__":
    main()

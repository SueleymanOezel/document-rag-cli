from pathlib import Path


def load_text_from_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

    if not path.is_file():
        raise ValueError(f"Pfad ist keine Datei: {file_path}")

    return path.read_text(encoding="utf-8")
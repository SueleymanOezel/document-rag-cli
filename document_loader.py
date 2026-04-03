from pathlib import Path

from pypdf import PdfReader


def load_text(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

    if not path.is_file():
        raise ValueError(f"Pfad ist keine Datei: {file_path}")

    return path.read_text(encoding="utf-8")


def load_pdf(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

    if not path.is_file():
        raise ValueError(f"Pfad ist keine Datei: {file_path}")

    reader = PdfReader(str(path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def load_text_from_file(file_path: str) -> str:
    # Rueckwaertskompatibel fuer bestehenden Code, der noch den alten Namen nutzt.
    return load_text(file_path)
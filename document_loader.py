from pathlib import Path
import re

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
    merged_text = "\n".join(pages_text)
    return _clean_pdf_text(merged_text)


def _clean_pdf_text(text: str) -> str:
    # 1) Repariert Silbentrennungen am Zeilenende.
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # 2) Ersetzt nur einzelne Zeilenumbrueche durch Leerzeichen und laesst Absaetze (\n\n) stehen.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 3) Reduziert mehrfache Leerzeichen auf genau eines.
    text = re.sub(r" +", " ", text)

    # 4) Absaetze bleiben erhalten, da Schritt 2 doppelte Zeilenumbrueche ausspart.
    return text.strip()


def load_text_from_file(file_path: str) -> str:
    # Rueckwaertskompatibel fuer bestehenden Code, der noch den alten Namen nutzt.
    return load_text(file_path)
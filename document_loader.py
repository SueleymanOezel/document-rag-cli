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
    # 1) Erzeugt vor erkannten Ueberschriften eine Absatzgrenze, damit Struktur erhalten bleibt.
    text = re.sub(r"\n([A-ZÄÖÜ][a-zäöüA-ZÄÖÜ\s\-()]{3,40})\n", r"\n\n## \1\n", text)

    # 2) Ersetzt einzelne Zeilenumbrueche durch Leerzeichen, laesst aber Absatzgrenzen (\n\n) unberuehrt.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 3) Bereinigt mehrfache Leerzeichen auf genau ein Leerzeichen.
    text = re.sub(r"  +", " ", text)

    # 4) Doppelte Zeilenumbrueche bleiben erhalten, da nur einzelne \n ersetzt werden.
    return text.strip()


def load_text_from_file(file_path: str) -> str:
    # Rueckwaertskompatibel fuer bestehenden Code, der noch den alten Namen nutzt.
    return load_text(file_path)
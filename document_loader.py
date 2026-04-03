from pathlib import Path
import re

from pypdf import PdfReader


def _clean_pdf_text(text: str) -> str:
    # Vereinheitlicht Zeilenenden aus unterschiedlichen Betriebssystemen.
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # Repariert Silbentrennungen am Zeilenende, z. B. "Syste-\nme" -> "Systeme".
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)

    # Merkt sich Absatzumbrueche, damit sie nach dem Cleaning erhalten bleiben.
    paragraph_marker = "__PARAGRAPH_BREAK__"
    cleaned = re.sub(r"\n{2,}", paragraph_marker, cleaned)

    # Ersetzt verbleibende einzelne Zeilenumbrueche durch Leerzeichen.
    cleaned = cleaned.replace("\n", " ")

    # Reduziert mehrere aufeinanderfolgende Leerzeichen/Tabs auf ein Leerzeichen.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    # Stellt Absatzumbrueche wieder als doppelte Zeilenumbrueche her.
    cleaned = cleaned.replace(paragraph_marker, "\n\n")

    return cleaned.strip()


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


def load_text_from_file(file_path: str) -> str:
    # Rueckwaertskompatibel fuer bestehenden Code, der noch den alten Namen nutzt.
    return load_text(file_path)
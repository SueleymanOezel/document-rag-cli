# Document RAG

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web--App-FF4B4B)
![Gemini](https://img.shields.io/badge/Gemini-2.0--flash--lite-4285F4)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Live](https://img.shields.io/badge/Live-Streamlit%20Cloud-success)

> Lade PDFs hoch und stelle Fragen dazu - mit oder ohne Gemini API-Key.  
> **[▶ Live Demo öffnen](https://document-rag-so.streamlit.app/)**

## Features

- 📄 Mehrere PDFs gleichzeitig hochladen und durchsuchen
- 🎯 Relevanz-Score pro gefundenem Abschnitt (farbig)
- 📁 Quellenangabe welches Dokument den Treffer enthält
- 💬 KI-Zusammenfassung mit eigenem Gemini API-Key (optional)
- 🔍 Retrieval-Modus ohne API-Key nutzbar
- 🕐 Chatverlauf mit Uhrzeit
- 🔄 Automatischer Retry bei API-Überlastung

## Demo

<!-- Screenshot hier einfügen -->
> Tipp: Screenshot der App machen und als
> assets/screenshot.png im Repo speichern,
> dann ![Screenshot](assets/screenshot.png) eintragen.

## Architektur

Die Pipeline der App:

```text
PDF Upload
  -> document_loader.py (PyMuPDF + Text-Cleaning)
  -> text_chunker.py (RecursiveCharacterTextSplitter, Markdown-aware)
  -> vector_store.py (SentenceTransformer + FAISS, Kurz-Chunk-Filter)
  -> optional qa_engine.py (Gemini API, Query Expansion)
  -> app.py (Streamlit UI)
```

## Lokale Installation

### Schritt 1: Repository klonen

```bash
git clone https://github.com/SueleymanOezel/document-rag-cli.git
cd document-rag-cli
```

### Schritt 2: Virtuelle Umgebung erstellen und Abhängigkeiten installieren

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### Schritt 3: Streamlit-App starten

```bash
streamlit run app.py
```

Kein API-Key nötig für den Retrieval-Modus.

## API-Key

- Optional: Gemini API-Key direkt in der App eingeben
- Kostenlosen Key gibt es auf https://aistudio.google.com
- Ohne Key: Retrieval-Modus (zeigt relevante Textabschnitte)
- Mit Key: volle KI-Antworten

## Bekannte Einschränkungen

- Im Gemini Free Tier sind bei hoher Last 503-Fehler möglich (die App versucht automatisch 3x)
- Sehr kurze Chunks (<50 Zeichen) werden automatisch herausgefiltert
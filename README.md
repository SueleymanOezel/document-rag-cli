# document-rag-cli

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gemini](https://img.shields.io/badge/Gemini-2.0--flash--lite-4285F4)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

> Frage deine Dokumente per CLI ab: schnell, lokal indexiert und optional mit Gemini-Antworten.

## 1. Kurzbeschreibung

`document-rag-cli` ist ein schlankes Python-CLI-Tool fuer Retrieval-Augmented Generation (RAG) auf Basis von Texten oder Textdateien. Es zerlegt Inhalte in Chunks, erzeugt Embeddings, sucht relevante Textstellen mit FAISS und beantwortet darauf basierende Fragen mit Gemini. Alternativ kann das Tool auch nur Retrieval ausgeben, ohne LLM-Antwort zu erzeugen.

## 2. Architektur-Uebersicht

```text
Eingabe
	|-- --text "..."
	'-- --file datei.txt
				 |
				 v
 [document_loader.py]  (nur bei --file)
				 |
				 v
 [text_chunker.py]
	 RecursiveCharacterTextSplitter
				 |
				 v
 [vector_store.py]
	 SentenceTransformer -> Embeddings
	 FAISS IndexFlatL2   -> Vektorindex
				 |
				 v
 Suche relevante Chunks (--top-k)
				 |
				 +--> --mode retrieve
				 |      Ausgabe der Top-K Chunks
				 |
				 '--> --mode rag
								|
								v
						[qa_engine.py]
						Gemini (google-genai)
								|
								v
							Antwort
```

## 3. Voraussetzungen

- Python 3.10+ (empfohlen: 3.11 oder 3.12)
- Gemini API-Key als Umgebungsvariable `GEMINI_API_KEY`
- Installierte Bibliotheken (siehe `requirements.txt`):
	- `langchain-text-splitters`
	- `sentence-transformers`
	- `faiss-cpu`
	- `numpy`
	- `google-genai`
	- `python-dotenv`

## 4. Installation

### Schritt 1: Repository klonen

```bash
git clone https://github.com/SueleymanOezel/document-rag-cli.git
cd document-rag-cli
```

### Schritt 2: Virtuelle Umgebung erstellen

```bash
python -m venv .venv
```

### Schritt 3: Virtuelle Umgebung aktivieren

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Schritt 4: Abhaengigkeiten installieren

```bash
pip install -r requirements.txt
```

### Schritt 5: `.env` anlegen

Lege im Projektroot eine Datei `.env` an:

```env
GEMINI_API_KEY=DEIN_GEMINI_API_KEY
```

Hinweis: Die `.env` wird beim Start automatisch geladen.

Sicherheitshinweis: Die `.env` darf niemals committed werden und muss in `.gitignore` stehen. In diesem Repo ist `.env` bereits in der `.gitignore` eingetragen.

## 5. Verwendung

### CLI-Syntax

```bash
python main.py (--text "..." | --file PFAD) --query "..." [OPTIONEN]
```

### Alle Parameter

- `--text` (string): Direkter Eingabetext (alternativ zu `--file`)
- `--file` (string): Pfad zu einer UTF-8-Textdatei (alternativ zu `--text`)
- `--query` (string, erforderlich): Frage an den Inhalt
- `--mode` (`retrieve` | `rag`, default: `rag`): Nur Retrieval oder vollständige RAG-Antwort
- `--chunk-size` (int, default: `500`): Maximale Chunk-Groesse in Zeichen
- `--chunk-overlap` (int, default: `50`): Zeichen-Ueberlappung zwischen benachbarten Chunks
- `--top-k` (int, default: `3`): Anzahl der relevantesten Chunks fuer Ausgabe bzw. Kontext

### Beispiel mit `--text`

```bash
python main.py \
	--text "Paris ist die Hauptstadt von Frankreich. Berlin ist die Hauptstadt von Deutschland." \
	--query "Was ist die Hauptstadt von Deutschland?"
```

### Beispiel mit `--file`

```bash
python main.py --file beispiel.txt --query "Worum geht es im Dokument?"
```

### `--mode retrieve` vs `--mode rag`

Nur relevante Chunks ausgeben (ohne LLM-Antwort):

```bash
python main.py --file beispiel.txt --query "test" --mode retrieve --top-k 3
```

Vollstaendiges RAG (Retrieval + Gemini-Antwort):

```bash
python main.py --file beispiel.txt --query "test" --mode rag --top-k 3
```

### Erklaerung `--chunk-size` und `--chunk-overlap`

- Groesseres `--chunk-size`:
	- Weniger, aber laengere Chunks
	- Mehr Kontext pro Chunk, potenziell weniger praezise Treffer
- Kleineres `--chunk-size`:
	- Mehr, aber kuerzere Chunks
	- Haeufig praezisere Treffer, aber eventuell Kontextverlust
- Hoeheres `--chunk-overlap`:
	- Reduziert Informationsverlust an Chunk-Grenzen
	- Erhoeht Redundanz und Rechenaufwand

Beispiel:

```bash
python main.py \
	--file beispiel.txt \
	--query "Welche Hauptthemen werden genannt?" \
	--chunk-size 800 \
	--chunk-overlap 120
```

### Erklaerung `--top-k`

`--top-k` bestimmt, wie viele relevante Chunks aus dem Vektorindex geholt werden.

- Kleines `top-k` (z. B. `1-3`): fokussierter Kontext, schneller
- Groesseres `top-k` (z. B. `5-10`): breiterer Kontext, aber mehr Rauschen moeglich

Beispiel:

```bash
python main.py --file beispiel.txt --query "Nenne die wichtigsten Fakten" --top-k 5
```

## 6. Bekannte Einschraenkungen

- Python 3.14:
	- In der aktuellen LangChain/Pydantic-v1-Kompatibilitaet koennen Warnungen auftreten.
	- Das Projekt unterdrueckt bekannte Warnungen bereits, dennoch sind 3.11/3.12 aktuell stabiler.
- Free-Tier-Quota (Gemini API):
	- Bei haeufigen Requests kann ein `429 ResourceExhausted` auftreten.
	- In diesem Fall kurz warten und erneut versuchen, oder auf ein bezahltes Kontingent wechseln.
- Aktuell werden nur UTF-8-Textdateien unterstuetzt.

## 7. Naechste geplante Features

- PDF-Support (Parsing und Chunking von PDF-Dokumenten)
- Einfache Web-UI fuer Upload, Fragen und Ergebnisanzeige
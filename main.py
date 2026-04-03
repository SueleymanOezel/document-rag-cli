import os
import warnings

# Unterdrückt störende Transformers-Loader-Ausgaben (u. a. "BertModel LOAD REPORT ...").
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Unterdrückt die Pydantic-v1-Kompatibilitätswarnung unter Python 3.14 aus langchain_core.
warnings.filterwarnings(
    "ignore",
    message=r".*Core Pydantic V1 functionality isn't compatible with Python 3\.14.*",
)

# Unterdrückt die HF-Hub-Hinweiswarnung zu nicht authentifizierten Requests.
warnings.filterwarnings(
    "ignore",
    message=r".*You are sending unauthenticated requests to the HF Hub.*",
)

# Unterdrückt die "BertModel LOAD REPORT ... embeddings.position_ids | UNEXPECTED"-Meldung.
warnings.filterwarnings(
    "ignore",
    message=r".*BertModel LOAD REPORT.*embeddings\.position_ids\s*\|\s*UNEXPECTED.*",
)

import argparse
from pathlib import Path

from dotenv import load_dotenv
from google.api_core.exceptions import NotFound, ResourceExhausted

from document_loader import load_pdf, load_text
from text_chunker import split_text_into_chunks
from vector_store import VectorStore
from qa_engine import setup_gemini, answer_question


def is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "resource_exhausted" in message
        or "quota" in message
        or ("429" in message and "gemini" in message)
        or "you exceeded your current quota" in message
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document-rag-cli",
        description="Ein einfaches RAG Q&A Tool für Textdateien oder direkten Texteingang.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument("--text", type=str, help="Direkter Texteingang als String.")
    input_group.add_argument("--file", type=str, help="Pfad zu einer .txt- oder .pdf-Datei.")

    parser.add_argument("--chunk-size", type=int, default=500, help="Maximale Größe eines Chunks in Zeichen.")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Überlappung zwischen zwei benachbarten Chunks in Zeichen.")
    parser.add_argument("--query", type=str, required=True, help="Frage an das Dokument.")
    parser.add_argument("--top-k", type=int, default=3, help="Anzahl der relevanten Chunks.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["retrieve", "rag"],
        default="rag",
        help="Ausführungsmodus: nur Retrieval oder vollständiges RAG.",
    )
    return parser


def get_input_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        file_path = Path(args.file)
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return load_text(args.file)
        if suffix == ".pdf":
            return load_pdf(args.file)
        raise ValueError("Nur .txt und .pdf Dateien werden unterstützt.")
    raise ValueError("Es wurde weder --text noch --file übergeben.")


def main() -> None:
    env_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=env_path, override=True)

    parser = build_parser()
    args = parser.parse_args()

    try:
        text = get_input_text(args)
        chunks = split_text_into_chunks(
            text=text,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        store = VectorStore()
        store.build_index(chunks)

        relevant_chunks = store.search(
            query=args.query,
            top_k=args.top_k,
        )

        if args.mode == "retrieve":
            print(f"Gefundene Chunks (Top {args.top_k}):")
            for idx, chunk in enumerate(relevant_chunks, start=1):
                print(f"[{idx}] {chunk}")
            return

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Fehler: GEMINI_API_KEY nicht gesetzt.\n"
                "Lege eine .env-Datei an oder setze die Umgebungsvariable."
            )

        model = setup_gemini(api_key=api_key)
        answer = answer_question(
            model=model,
            question=args.query,
            context_chunks=relevant_chunks,
        )

    except FileNotFoundError as exc:
        # Spezifischer Dateifehler mit Pfad, damit sofort klar ist, welche Eingabedatei fehlt.
        print(f"Fehler: {exc}")
        raise SystemExit(1)
    except ResourceExhausted:
        # 429 von Gemini: Quota/Rate-Limit wurde erreicht und sollte als klare Aktion kommuniziert werden.
        print("Fehler: Gemini API-Quota überschritten. Warte kurz und versuche es erneut.")
        raise SystemExit(1)
    except NotFound:
        # 404 vom Modell-Endpunkt: meist ist der Modellname ungültig oder nicht verfügbar.
        print("Fehler: Das Gemini-Modell wurde nicht gefunden.\nPrüfe den Modellnamen in qa_engine.py.")
        raise SystemExit(1)
    except EnvironmentError as exc:
        # Konfigurationsfehler wie fehlender API-Key werden separat und benutzerfreundlich ausgegeben.
        print(exc)
        raise SystemExit(1)
    except ValueError as exc:
        # Validierungsfehler aus Input-Parsing oder Antwortverarbeitung.
        print(f"Fehler: {exc}")
        raise SystemExit(1)
    except UnicodeDecodeError:
        # Dateiinhalt ist kein UTF-8 und kann deshalb nicht verarbeitet werden.
        print("Fehler: Die Datei konnte nicht als UTF-8 gelesen werden.")
        raise SystemExit(1)
    except Exception as exc:
        if is_quota_error(exc):
            print("Fehler: Gemini API-Quota überschritten. Warte kurz und versuche es erneut.")
            raise SystemExit(1)
        # Fallback für alle nicht explizit behandelten Fehler.
        print(f"Unerwarteter Fehler: {exc}")
        raise SystemExit(1)

    print("Antwort:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from document_loader import load_text_from_file
from text_chunker import split_text_into_chunks
from vector_store import VectorStore
from qa_engine import setup_gemini, answer_question


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document-rag-cli",
        description="Ein einfaches RAG Q&A Tool für Textdateien oder direkten Texteingang.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument("--text", type=str, help="Direkter Texteingang als String.")
    input_group.add_argument("--file", type=str, help="Pfad zu einer UTF-8-Textdatei.")

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
        return load_text_from_file(args.file)
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
            print("Fehler: GEMINI_API_KEY ist nicht gesetzt.")
            raise SystemExit(1)

        model = setup_gemini(api_key=api_key)
        answer = answer_question(
            model=model,
            question=args.query,
            context_chunks=relevant_chunks,
        )

    except FileNotFoundError as exc:
        print(f"Fehler: {exc}")
        raise SystemExit(1)
    except ValueError as exc:
        print(f"Fehler: {exc}")
        raise SystemExit(1)
    except UnicodeDecodeError:
        print("Fehler: Die Datei konnte nicht als UTF-8 gelesen werden.")
        raise SystemExit(1)
    except Exception as exc:
        print(f"Unerwarteter Fehler: {exc}")
        raise SystemExit(1)

    print("Antwort:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
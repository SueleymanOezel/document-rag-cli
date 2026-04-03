import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from document_loader import load_pdf, load_text
from qa_engine import answer_question, setup_gemini
from text_chunker import split_text_into_chunks
from vector_store import VectorStore


# Laedt Variablen aus der .env, damit GEMINI_API_KEY verfuegbar ist.
load_dotenv()

st.set_page_config(page_title="Document RAG", page_icon="📄", layout="wide")
st.title("📄 Document RAG")


def init_session_state() -> None:
    # Haltet den bereits gebauten Index zwischen Interaktionen fest.
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "index_signature" not in st.session_state:
        st.session_state.index_signature = None


def read_uploaded_document(uploaded_file) -> str:
    # Bestehende Loader erwarten einen Dateipfad; daher speichern wir den Upload kurz in eine Temp-Datei.
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = Path(tmp_file.name)

    try:
        if suffix == ".pdf":
            return load_pdf(str(temp_path))
        if suffix == ".txt":
            return load_text(str(temp_path))
        raise ValueError("Nur .txt und .pdf Dateien werden unterstützt.")
    finally:
        if temp_path.exists():
            temp_path.unlink()


def compute_signature(source_text: str, chunk_size: int, chunk_overlap: int) -> str:
    # Signatur entscheidet, ob ein Re-Indexing noetig ist.
    payload = f"{chunk_size}|{chunk_overlap}|{source_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_or_reuse_index(source_text: str, chunk_size: int, chunk_overlap: int) -> None:
    signature = compute_signature(source_text, chunk_size, chunk_overlap)

    if st.session_state.index_signature == signature and st.session_state.vector_store is not None:
        return

    with st.spinner("Lade und indexiere Dokument..."):
        chunks = split_text_into_chunks(
            source_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        store = VectorStore()
        store.build_index(chunks)

    st.session_state.vector_store = store
    st.session_state.chunks = chunks
    st.session_state.index_signature = signature


init_session_state()

with st.sidebar:
    st.header("Einstellungen")
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1500, value=500, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, value=50, step=10)
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=3, step=1)
    mode = st.radio("Modus", options=["RAG (mit Gemini)", "Nur Retrieval"])

tab_upload, tab_text = st.tabs(["Datei-Upload", "Direkter Text"])

with tab_upload:
    uploaded_file = st.file_uploader("Dokument hochladen (.txt, .pdf)", type=["txt", "pdf"])

with tab_text:
    direct_text = st.text_area("Oder Text direkt einfuegen", height=220)

question = st.text_input("Deine Frage")
ask_clicked = st.button("Fragen", type="primary")

if ask_clicked:
    try:
        if not question.strip():
            raise ValueError("Bitte gib eine Frage ein.")

        source_text = ""
        if uploaded_file is not None:
            source_text = read_uploaded_document(uploaded_file)
        elif direct_text.strip():
            source_text = direct_text

        if not source_text.strip():
            raise ValueError("Bitte lade eine Datei hoch oder gib direkten Text ein.")

        build_or_reuse_index(source_text, chunk_size, chunk_overlap)

        with st.spinner("Suche relevante Abschnitte..."):
            retrieved_chunks = st.session_state.vector_store.search(question, top_k=top_k)

        st.subheader("Gefundene Abschnitte")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            with st.expander(f"📄 Abschnitt {i}", expanded=False):
                with st.container(border=True):
                    # Codeblock in Markdown sorgt fuer eine gut lesbare, code-aehnliche Darstellung.
                    safe_chunk = chunk.replace("```", "'''")
                    st.markdown(f"```text\n{safe_chunk}\n```")

        if mode == "Nur Retrieval":
            st.success("Retrieval abgeschlossen.")
        else:
            try:
                api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                api_key = None
            if not api_key:
                import os

                api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise ValueError("GEMINI_API_KEY nicht gefunden. Bitte in .env setzen.")

            with st.spinner("Gemini denkt nach..."):
                client = setup_gemini(api_key)
                response = answer_question(client, question, retrieved_chunks)

            st.divider()
            st.caption(f"Basierend auf {len(retrieved_chunks)} gefundenen Abschnitten")
            st.success(response)

    except Exception as exc:
        st.error(f"Fehler: {exc}")
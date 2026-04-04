import hashlib
import html
from datetime import datetime
import tempfile
from pathlib import Path

import streamlit as st

from document_loader import load_pdf, load_text
from qa_engine import QAEngine
from text_chunker import split_text_into_chunks
from vector_store import VectorStore

st.set_page_config(page_title="Document RAG", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
        .stApp {
            background: #f7f7f5;
        }
        [data-testid="stSidebar"] {
            background: #eeeeea;
            border-right: 1px solid #d8d8d3;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            background: #eeeeea;
        }
        .block-container {
            max-width: 860px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            background: #ffffff;
            border: 1px solid #e6e6e2;
            border-radius: 14px;
            padding-left: 1.4rem;
            padding-right: 1.4rem;
        }
        h1, h2, h3 {
            letter-spacing: -0.01em;
            color: #1f1f1d;
        }
        .stTextInput > div > div > input,
        .stTextArea textarea {
            background-color: #fcfcfb;
            color: #1f1f1d;
            border: 1px solid #bfc0b8;
            border-radius: 10px;
        }
        .stTextInput > div > div > input:focus,
        .stTextArea textarea:focus {
            border-color: #8f9086;
            box-shadow: 0 0 0 1px #8f9086;
        }
        .stButton button {
            border-radius: 10px;
            border: 1px solid #a9aa9f;
            background: #f2f2ef;
            color: #1f1f1d;
            font-weight: 600;
        }
        .stButton button:hover {
            background: #e7e7e2;
            border-color: #8f9086;
        }
        .stButton button:focus {
            box-shadow: 0 0 0 1px #8f9086;
        }
        [data-testid="stExpander"] {
            border: 1px solid #e4e4e0;
            border-radius: 12px;
            background: #ffffff;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    # Haltet den bereits gebauten Index zwischen Interaktionen fest.
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "chunk_counts" not in st.session_state:
        st.session_state.chunk_counts = {}
    if "index_signature" not in st.session_state:
        st.session_state.index_signature = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""


def render_api_key_input() -> None:
    has_key = bool(st.session_state.get("gemini_api_key", "").strip())
    with st.expander("🔑 Gemini API-Key (optional)", expanded=not has_key):
        st.text_input("API-Key", type="password", key="gemini_api_key")
        st.caption("Deinen kostenlosen Key bekommst du auf aistudio.google.com")
        st.info(
            "Ohne API-Key werden nur die relevanten Textabschnitte angezeigt "
            "(Retrieval-Modus). Mit Key erhältst du eine KI-Zusammenfassung."
        )


def add_chat_history_entry(question: str, answer: str) -> None:
    entry = {
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().strftime("%H:%M"),
    }
    st.session_state.chat_history.append(entry)
    st.session_state.chat_history = st.session_state.chat_history[-10:]


def render_chat_history() -> None:
    if not st.session_state.chat_history:
        return

    for index, entry in enumerate(st.session_state.chat_history):
        question = html.escape(entry["question"])
        answer = html.escape(entry["answer"]).replace("\n", "<br>")

        left, right = st.columns([0.88, 0.12])
        left.markdown(f"**{question}**")
        right.markdown(
            f"<div style='text-align: right; color: #666;'>{entry['timestamp']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='margin-left: 1.25rem;'>{answer}</div>",
            unsafe_allow_html=True,
        )

        if index < len(st.session_state.chat_history) - 1:
            st.divider()


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


def compute_signature(documents: list[tuple[str, str]], chunk_size: int, chunk_overlap: int) -> str:
    # Signatur entscheidet, ob ein Re-Indexing noetig ist.
    parts = [f"{name}:{text}" for name, text in documents]
    payload = f"{chunk_size}|{chunk_overlap}|{'||'.join(parts)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_or_reuse_index(documents: list[tuple[str, str]], chunk_size: int, chunk_overlap: int) -> bool:
    signature = compute_signature(documents, chunk_size, chunk_overlap)

    if st.session_state.index_signature == signature and st.session_state.vector_store is not None:
        return False

    progress_placeholder = st.empty()
    chunks_pro_pdf: list[list[tuple[str, str]]] = []
    chunk_counts: dict[str, int] = {}

    for filename, source_text in documents:
        progress_placeholder.info(f"📄 Verarbeite {filename}...")
        chunks = split_text_into_chunks(
            source_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks_mit_quelle = [(chunk, filename) for chunk in chunks]
        chunks_pro_pdf.append(chunks_mit_quelle)
        chunk_counts[filename] = len(chunks_mit_quelle)

    store = VectorStore()
    store.build_index(chunks_pro_pdf)

    total_chunks = sum(chunk_counts.values())
    progress_placeholder.success(
        f"✅ {len(documents)} Dokumente indexiert: {total_chunks} Chunks gesamt"
    )

    st.session_state.vector_store = store
    st.session_state.chunks = store.chunks
    st.session_state.chunk_counts = chunk_counts
    st.session_state.index_signature = signature
    return True


init_session_state()

st.title("📄 Document RAG")
st.caption("Lade PDFs hoch und stelle Fragen – mit oder ohne Gemini API-Key.")
st.divider()

render_api_key_input()

with st.sidebar:
    st.header("Einstellungen")
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1500, value=500, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, value=50, step=10)
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=3, step=1)

st.subheader("📂 Dokumente")
uploaded_files = st.file_uploader("PDFs hochladen", type="pdf", accept_multiple_files=True)

st.subheader("💬 Frage stellen")
question = st.text_input("Deine Frage")
ask_clicked = st.button("Fragen", type="primary")

if ask_clicked:
    try:
        if not question.strip():
            raise ValueError("Bitte gib eine Frage ein.")

        documents: list[tuple[str, str]] = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                source_text = read_uploaded_document(uploaded_file)
                if source_text.strip():
                    documents.append((uploaded_file.name, source_text))

        if not documents:
            raise ValueError("Bitte lade mindestens eine PDF hoch.")

        index_rebuilt = build_or_reuse_index(documents, chunk_size, chunk_overlap)
        if index_rebuilt:
            for filename, count in st.session_state.chunk_counts.items():
                st.write(f"✅ {filename} ({count} Chunks)")

        with st.spinner("Suche relevante Abschnitte..."):
            retrieved_chunks = st.session_state.vector_store.search(question, top_k=top_k)

        st.subheader("🔍 Gefundene Abschnitte")
        for i, (chunk_tuple, score) in enumerate(retrieved_chunks, start=1):
            chunk, filename = chunk_tuple
            if score >= 80:
                score_color = "#2d9e5e"
                score_suffix = ""
            elif score >= 50:
                score_color = "#d97706"
                score_suffix = ""
            else:
                score_color = "#dc2626"
                score_suffix = " (möglicherweise nicht relevant)"

            with st.container(border=True):
                st.markdown(
                    (
                        f"📄 Abschnitt {i} · {len(chunk)} Zeichen · "
                        f"📁 {filename} · "
                        f"<span style='color: {score_color}; font-weight: 700;'>"
                        f"🎯 {score}% Relevanz{score_suffix}</span>"
                    ),
                    unsafe_allow_html=True,
                )
                # Codeblock in Markdown sorgt fuer eine gut lesbare, code-aehnliche Darstellung.
                safe_chunk = chunk.replace("```", "'''")
                st.markdown(f"```text\n{safe_chunk}\n```")

        hat_key = bool(st.session_state.get("gemini_api_key", "").strip())

        if hat_key:
            with st.spinner("Gemini denkt nach..."):
                qa_engine = QAEngine(api_key=st.session_state.gemini_api_key)
                response = qa_engine.answer_question(
                    question=question,
                    context_chunks=[chunk for (chunk, _), _ in retrieved_chunks],
                )

            st.subheader("💡 Antwort")
            st.info(response)
            answer = response
        else:
            answer = "Retrieval-Modus: Relevante Abschnitte wurden angezeigt."
            st.subheader("💡 Antwort")
            st.info("Füge einen Gemini API-Key hinzu, um eine KI-Zusammenfassung zu erhalten.")

        add_chat_history_entry(question=question, answer=answer)

    except Exception as exc:
        st.error(f"Fehler: {exc}")

st.divider()
st.subheader("🕐 Chatverlauf")
render_chat_history()
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
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""


def render_api_key_input() -> None:
    st.subheader("Gemini API-Key")
    st.text_input("API-Key (optional)", type="password", key="gemini_api_key")
    st.caption("Deinen kostenlosen Key bekommst du auf aistudio.google.com")
    st.info(
        "ℹ️ Ohne API-Key werden nur die relevanten Textabschnitte angezeigt "
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

    st.subheader("Chatverlauf")
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


def compute_signature(source_text: str, chunk_size: int, chunk_overlap: int) -> str:
    # Signatur entscheidet, ob ein Re-Indexing noetig ist.
    payload = f"{chunk_size}|{chunk_overlap}|{source_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_or_reuse_index(source_text: str, chunk_size: int, chunk_overlap: int) -> bool:
    signature = compute_signature(source_text, chunk_size, chunk_overlap)

    if st.session_state.index_signature == signature and st.session_state.vector_store is not None:
        return False

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
    return True


init_session_state()
render_api_key_input()

with st.sidebar:
    st.header("Einstellungen")
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1500, value=500, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, value=50, step=10)
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=3, step=1)

tab_upload, tab_text = st.tabs(["Datei-Upload", "Direkter Text"])

with tab_upload:
    uploaded_file = st.file_uploader("Dokument hochladen (.txt, .pdf)", type=["txt", "pdf"])

with tab_text:
    direct_text = st.text_area("Oder Text direkt einfuegen", height=220)

history_container = st.container()

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

        index_rebuilt = build_or_reuse_index(source_text, chunk_size, chunk_overlap)
        if index_rebuilt:
            st.info(f"📊 Dokument indexiert: {len(st.session_state.chunks)} Chunks erstellt")

        with st.spinner("Suche relevante Abschnitte..."):
            retrieved_chunks = st.session_state.vector_store.search(question, top_k=top_k)

        st.subheader("Gefundene Abschnitte")
        for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
            if score >= 80:
                score_color = "#2d9e5e"
                score_suffix = ""
            elif score >= 50:
                score_color = "#d97706"
                score_suffix = ""
            else:
                score_color = "#dc2626"
                score_suffix = " (möglicherweise nicht relevant)"

            with st.expander(f"Details zu Abschnitt {i}", expanded=False):
                st.markdown(
                    (
                        f"📄 Abschnitt {i} · {len(chunk)} Zeichen · "
                        f"<span style='color: {score_color}; font-weight: 700;'>"
                        f"🎯 {score}% Relevanz{score_suffix}</span>"
                    ),
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    # Codeblock in Markdown sorgt fuer eine gut lesbare, code-aehnliche Darstellung.
                    safe_chunk = chunk.replace("```", "'''")
                    st.markdown(f"```text\n{safe_chunk}\n```")

        hat_key = bool(st.session_state.get("gemini_api_key", "").strip())

        if hat_key:
            with st.spinner("Gemini denkt nach..."):
                qa_engine = QAEngine(api_key=st.session_state.gemini_api_key)
                response = qa_engine.answer_question(
                    question=question,
                    context_chunks=[chunk for chunk, _ in retrieved_chunks],
                )

            st.divider()
            st.subheader("💬 Antwort")
            st.markdown(response)
            answer = response
        else:
            st.info("💡 Füge einen Gemini API-Key hinzu um eine KI-Zusammenfassung zu erhalten.")
            answer = "Retrieval-Modus: Relevante Abschnitte wurden angezeigt."

        add_chat_history_entry(question=question, answer=answer)

    except Exception as exc:
        st.error(f"Fehler: {exc}")

with history_container:
    render_chat_history()
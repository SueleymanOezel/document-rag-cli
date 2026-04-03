import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: list[tuple[str, str]] = []

    def build_index(self, chunks_pro_pdf: list[list[tuple[str, str]]]) -> None:
        if not chunks_pro_pdf:
            raise ValueError("Es wurden keine Chunks zum Indexieren übergeben.")

        alle_chunks: list[tuple[str, str]] = []
        for chunks in chunks_pro_pdf:
            alle_chunks.extend(chunks)

        gefilterte_chunks = [
            (chunk_text, dateiname)
            for chunk_text, dateiname in alle_chunks
            if len(chunk_text.strip()) >= 50
        ]

        if not gefilterte_chunks:
            raise ValueError("Nach dem Filtern sind keine Chunks mit mindestens 50 Zeichen übrig.")

        self.chunks = gefilterte_chunks

        embeddings = self.model.encode([chunk_text for chunk_text, _ in gefilterte_chunks])
        embeddings = np.array(embeddings, dtype="float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> list[tuple[tuple[str, str], int]]:
        if self.index is None:
            raise ValueError("Der FAISS-Index wurde noch nicht erstellt.")

        if not query.strip():
            raise ValueError("Die Suchanfrage ist leer.")

        if top_k <= 0:
            raise ValueError("top_k muss größer als 0 sein.")

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype="float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results: list[tuple[tuple[str, str], int]] = []
        for distance, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                score = round(max(0, 100 - (float(distance) * 15)))
                results.append((self.chunks[idx], score))

        return results
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: list[str] = []

    def build_index(self, chunks: list[str]) -> None:
        if not chunks:
            raise ValueError("Es wurden keine Chunks zum Indexieren übergeben.")

        self.chunks = chunks

        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings, dtype="float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        if self.index is None:
            raise ValueError("Der FAISS-Index wurde noch nicht erstellt.")

        if not query.strip():
            raise ValueError("Die Suchanfrage ist leer.")

        if top_k <= 0:
            raise ValueError("top_k muss größer als 0 sein.")

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype="float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results: list[str] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path: str, meta_path: str, embedding_model: str):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.model = SentenceTransformer(embedding_model)
        self.top_k = 3

    def embed(self, text: str):
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")

    def retrieve(self, query: str, top_k: int = None):
        if top_k is None:
            top_k = self.top_k
        q = self.embed(query).reshape(1, -1)
        sims, ids = self.index.search(q, top_k)
        out = []
        for idx, sim in zip(ids[0], sims[0]):
            if idx == -1:
                continue
            meta = self.meta[int(idx)]
            out.append({"source": meta["source"], "text": meta["text"], "score": float(sim)})
        return out

    def best_similarity(self, query: str) -> float:
        q = self.embed(query).reshape(1, -1)
        sims, _ = self.index.search(q, 1)
        return float(sims[0][0])

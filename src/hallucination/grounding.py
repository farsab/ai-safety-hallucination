from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

class Grounding:
    def __init__(self, retriever, min_sim: float = 0.3):
        # Reuse retriever's embedding model to avoid extra downloads
        self.encoder = retriever.model
        self.min_sim = min_sim

    def grounded_similarity(self, answer: str, ctx: List[Dict]) -> float:
        if not ctx:
            return 0.0
        a = self.encoder.encode([answer], convert_to_numpy=True, normalize_embeddings=True)[0]
        c = [self.encoder.encode([c["text"]], convert_to_numpy=True, normalize_embeddings=True)[0] for c in ctx]
        c_mat = np.vstack(c).astype("float32")
        sim = float(np.max(a @ c_mat.T))  # cosine with normalized
        return sim

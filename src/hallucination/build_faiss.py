import argparse
import os, json, glob
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def read_corpus(corpus_dir: str):
    docs = []
    for path in glob.glob(os.path.join(corpus_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                docs.append({"source": os.path.relpath(path, start=".").replace("\\", "/"), "text": text})
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Folder with .txt files")
    ap.add_argument("--out", required=True, help="Path to save FAISS index")
    ap.add_argument("--meta", required=True, help="Path to save metadata JSON")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    docs = read_corpus(args.corpus)
    if not docs:
        raise SystemExit("No .txt files found in corpus.")

    model = SentenceTransformer(args.model)
    embeddings = model.encode([d["text"] for d in docs], convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, args.out)
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"Built index with {len(docs)} docs → {args.out}")
    print(f"Saved metadata → {args.meta}")

if __name__ == "__main__":
    main()

# AI Safety & Hallucination-Aware RAG

A minimal, local-first pipeline that demonstrates **AI safety** and **hallucination avoidance** patterns:

- Retrieval-Augmented Generation (FAISS + Sentence-Transformers)
- Prompt-injection heuristics
- Toxic content filtering
- OOD (query-to-corpus similarity) gate
- **Confidence estimation** from token probabilities (refuse on low confidence)
- **Grounding check** (semantic similarity of answer to retrieved context)

> Works offline after models are downloaded once. Replace models in `config.yaml` as needed.

---

## Quickstart

```bash
# 1) Create venv (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Build FAISS index over `data/corpus/*.txt`
python -m src.hallucination.build_faiss --corpus data/corpus --out data/index.faiss --meta data/meta.json

# 4) Ask a question
python main.py --query "What is retrieval-augmented generation and how does it reduce hallucinations?"
```

### Expected behavior
- If your query is **too far** from the corpus → graceful refusal (OOD gate).
- If the model’s **confidence** in its own tokens is low → refusal.
- If the answer is **not grounded** in the retrieved context → refusal.
- If output is **toxic** → blocked.

---

## Repo Layout

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── data/
│   └── corpus/          # sample text files (you can replace with your own)
├── src/
│   ├── config_loader.py
│   ├── model_wrapper.py
│   ├── pipeline.py
│   ├── safety/
│   │   ├── toxic_filter.py
│   │   └── prompt_injection_detector.py
│   └── hallucination/
│       ├── build_faiss.py
│       ├── retrieval.py
│       ├── grounding.py
│       └── confidence_estimator.py
└── tests/
    └── test_pipeline.py
```

### Notes
- Default models in `config.yaml` are small and convenient for demo. Swap to larger ones for better quality.
- **Citations** printed by `main.py` reflect the retrieved context files.
- This is a teaching repo; adapt guardrails to your product’s risk model.

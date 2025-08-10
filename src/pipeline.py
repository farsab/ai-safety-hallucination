from typing import List, Dict, Any
from src.config_loader import load_config
from src.model_wrapper import ModelWrapper
from src.safety.toxic_filter import ToxicFilter
from src.safety.prompt_injection_detector import PromptInjectionDetector
from src.hallucination.retrieval import Retriever
from src.hallucination.grounding import Grounding
from src.hallucination.confidence_estimator import OODGate

class SafetyPipeline:
    def __init__(self):
        self.config = load_config()
        self.model = ModelWrapper(self.config["model"]["name"])

        # Safety modules
        self.toxic_filter = ToxicFilter(self.config["filters"]["toxic_threshold"])
        self.prompt_injection_detector = PromptInjectionDetector()

        # Retrieval & semantics
        self.retriever = Retriever(
            index_path=self.config["retrieval"]["faiss_index_path"],
            meta_path=self.config["retrieval"]["meta_path"],
            embedding_model=self.config["retrieval"]["embedding_model"],
        )
        self.ood_gate = OODGate(self.retriever, self.config["retrieval"]["min_query_sim"])
        self.grounding = Grounding(self.retriever, min_sim=self.config["grounding"]["min_sim"])

    def build_prompt(self, user_prompt: str, context_chunks: List[Dict[str, str]]) -> str:
        ctx_text = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(context_chunks)])
        system = (
            "You are a careful, concise assistant. Use ONLY the provided context to answer. "
            "If the answer isn't clearly in the context, say you don't know. "
            "Cite sources like [1], [2] based on the context IDs."
        )
        return f"{system}\n\nContext:\n{ctx_text}\n\nUser: {user_prompt}\nAssistant:"

    def run(self, user_prompt: str) -> Dict[str, Any]:
        reasons = {}

        # 1) Prompt-injection heuristic
        if self.prompt_injection_detector.detect(user_prompt):
            return {
                "answer": "⚠️ Prompt rejected: possible injection attempt.",
                "citations": [],
                "reasons": {"prompt_injection": "triggered"},
            }

        # 2) OOD gate on the query
        q_sim = self.ood_gate.query_similarity(user_prompt)
        reasons["query_similarity"] = f"{q_sim:.2f}"
        if q_sim < self.ood_gate.threshold:
            return {
                "answer": "I don't have enough domain grounding to answer this safely. "
                          "Please provide more context or add relevant documents to the corpus.",
                "citations": [],
                "reasons": reasons,
            }

        # 3) Retrieve top-k context
        ctx = self.retriever.retrieve(user_prompt, top_k=self.retriever.top_k)

        # 4) Build grounded prompt and generate
        prompt = self.build_prompt(user_prompt, ctx)
        text, conf = self.model.generate_with_confidence(
            prompt,
            max_new_tokens=self.config["model"]["max_new_tokens"],
            temperature=self.config["model"]["temperature"],
        )
        reasons["confidence"] = f"{conf:.2f}"

        # 5) Confidence check
        if conf < self.config["confidence"]["min_confidence"]:
            return {
                "answer": "I'm not confident enough in my answer to respond reliably.",
                "citations": [{"source": c["source"]} for c in ctx],
                "reasons": reasons,
            }

        # 6) Grounding check
        gsim = self.grounding.grounded_similarity(text, ctx)
        reasons["grounding_similarity"] = f"{gsim:.2f}"
        if gsim < self.grounding.min_sim:
            return {
                "answer": "I can't verify this with the provided context, so I'll refrain from answering.",
                "citations": [{"source": c["source"]} for c in ctx],
                "reasons": reasons,
            }

        # 7) Toxicity check on output
        if self.toxic_filter.is_toxic(text):
            return {
                "answer": "⚠️ Response blocked due to toxic content.",
                "citations": [],
                "reasons": reasons,
            }

        return {
            "answer": text,
            "citations": [{"source": c["source"]} for c in ctx],
            "reasons": reasons,
        }

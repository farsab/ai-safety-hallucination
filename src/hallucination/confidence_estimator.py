class OODGate:
    def __init__(self, retriever, threshold: float):
        self.retriever = retriever
        self.threshold = threshold

    def query_similarity(self, query: str) -> float:
        return self.retriever.best_similarity(query)

import re

class PromptInjectionDetector:
    def __init__(self):
        self.injection_patterns = [
            r"ignore (all|previous) (instructions|rules)",
            r"system override",
            r"please leak (the )?(key|secrets|password)",
            r"delete all data",
            r"disable (safety|guardrails)",
        ]

    def detect(self, prompt: str) -> bool:
        return any(re.search(p, prompt, re.IGNORECASE) for p in self.injection_patterns)

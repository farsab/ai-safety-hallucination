from transformers import pipeline

class ToxicFilter:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        # This model returns labels like 'toxic'/'non-toxic' depending on the dataset
        self.detector = pipeline("text-classification", model="unitary/toxic-bert")

    def is_toxic(self, text: str) -> bool:
        result = self.detector(text)[0]
        label = result["label"].lower()
        score = result["score"]
        # Be conservative: if model outputs multiple labels (some versions do), pick 'toxic' path
        return ("toxic" in label) and (score >= self.threshold)

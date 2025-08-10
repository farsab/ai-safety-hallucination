from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

class ModelWrapper:
    """
    Thin wrapper around HF causal LM with a built-in token-probability confidence estimator.
    """
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def generate_with_confidence(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode text
        text = self.tokenizer.decode(out.sequences[0], skip_special_tokens=True)

        # Confidence from token probs of generated tokens
        scores = out.scores  # list of logits per step: len == generated tokens
        gen_ids = out.sequences[0][input_len:]  # ids for generated part

        if len(scores) == 0:
            return text, 0.0

        probs = []
        for step_logits, tok_id in zip(scores, gen_ids):
            step_prob = F.softmax(step_logits[0], dim=-1)[tok_id].item()
            probs.append(step_prob)

        # Mean probability as confidence
        conf = float(sum(probs) / max(1, len(probs)))
        return text, conf

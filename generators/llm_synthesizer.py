"""
LLM-powered text and labeled dataset synthesis.
Generates instruction-response pairs, sentiment-labeled text, and NLP corpora.
"""
from __future__ import annotations
import json
import time
import anthropic
from dataclasses import dataclass


@dataclass
class SynthesisConfig:
    domain: str = "customer_support"
    n_samples: int = 100
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 1.0
    labels: list[str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = ["positive", "negative", "neutral"]


class LLMSynthesizer:
    """
    Uses Claude to generate labeled text datasets.
    Supports: sentiment classification, intent detection, NER, Q&A pairs.
    """

    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.client = anthropic.Anthropic()

    def _generate_batch(self, label: str, n: int) -> list[dict]:
        prompt = f"""Generate {n} realistic {self.config.domain} text examples with label '{label}'.
Return ONLY a JSON array of objects with keys "text" and "label".
Examples should be varied in length, tone, and phrasing. No duplicates.
Domain context: {self.config.domain}
Label meaning: {label} sentiment/intent."""

        resp = self.client.messages.create(
            model=self.config.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    def generate(self) -> list[dict]:
        labels = self.config.labels
        per_label = self.config.n_samples // len(labels)
        all_samples = []
        for label in labels:
            print(f"  Generating {per_label} '{label}' examples...")
            batch = self._generate_batch(label, per_label)
            all_samples.extend(batch)
            time.sleep(0.5)
        print(f"Generated {len(all_samples)} labeled examples")
        return all_samples

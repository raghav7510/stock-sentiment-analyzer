"""FinBERT inference; aggregation is kept dependency-light in metrics.py."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .metrics import aggregate_sentiment


@dataclass(frozen=True)
class SentimentResult:
    label: str
    score: float
    confidence: float
    probabilities: dict[str, float]


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _class_names(model) -> list[str]:
    """Read class order from model metadata; never assume logit index order."""
    mapping = model.config.id2label
    names = []
    for index in range(model.config.num_labels):
        raw = str(mapping.get(index, mapping.get(str(index), ""))).lower()
        label = next((name for name in ("positive", "negative", "neutral") if name in raw), None)
        if label is None:
            raise ValueError(f"Unsupported FinBERT class label at index {index}: {raw!r}")
        names.append(label)
    if set(names) != {"positive", "negative", "neutral"}:
        raise ValueError(f"Expected positive/negative/neutral classes, got {names}")
    return names


def analyse_text(text: str, model_name: str = "ProsusAI/finbert") -> SentimentResult:
    cleaned = " ".join((text or "").split())
    if len(cleaned) < 3:
        raise ValueError("Text must contain at least three characters.")
    tokenizer, model = _load_model(model_name)
    encoded = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=256)
    with torch.inference_mode():
        probs = torch.softmax(model(**encoded).logits, dim=-1)[0].cpu().tolist()
    class_names = _class_names(model)
    probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    score = probabilities["positive"] - probabilities["negative"]
    label = max(probabilities, key=probabilities.__getitem__)
    return SentimentResult(label, score, max(probabilities.values()), probabilities)

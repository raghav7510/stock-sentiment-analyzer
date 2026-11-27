"""FinBERT inference and confidence-aware sentiment aggregation."""
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
import math
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = {0: "negative", 1: "neutral", 2: "positive"}

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

def analyse_text(text: str, model_name: str = "ProsusAI/finbert") -> SentimentResult:
    cleaned = " ".join((text or "").split())
    if len(cleaned) < 3:
        raise ValueError("Text must contain at least three characters.")
    tokenizer, model = _load_model(model_name)
    encoded = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=256)
    with torch.inference_mode():
        probs = torch.softmax(model(**encoded).logits, dim=-1)[0].cpu().tolist()
    probabilities = {LABELS[i]: float(probs[i]) for i in range(3)}
    score = probabilities["positive"] - probabilities["negative"]
    return SentimentResult(LABELS[max(range(3), key=probs.__getitem__)], score, max(probabilities.values()), probabilities)

def aggregate_sentiment(items: Iterable[dict]) -> dict:
    """Aggregate scored articles with optional positive weights."""
    rows = list(items)
    if not rows:
        return {"score": 0.0, "article_count": 0, "coverage": "no_data"}
    weighted = []
    for row in rows:
        score, weight = float(row["score"]), float(row.get("weight", 1.0))
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError("Sentiment scores must be finite and between -1 and 1.")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Article weights must be finite and positive.")
        weighted.append((score, weight))
    total_weight = sum(w for _, w in weighted)
    return {"score": round(sum(s*w for s, w in weighted) / total_weight, 4), "article_count": len(rows), "coverage": "scored"}

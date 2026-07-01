"""Transparent local TF-IDF retrieval; no extra embedding model is required."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


_TOKEN = re.compile(r"[a-z][a-z0-9'-]{1,}", re.I)
_STOP = {
    "and", "are", "for", "from", "has", "have", "into", "its", "that", "the",
    "their", "this", "was", "were", "will", "with", "within", "timor", "leste",
}


def tokenise(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text) if token.lower() not in _STOP]


def retrieve_chunks(query: str, chunks: Iterable[dict], top_k: int = 6) -> list[dict]:
    """Rank chunks with cosine similarity over an in-memory TF-IDF representation."""

    rows = list(chunks)
    if not rows or top_k <= 0:
        return []
    documents = [Counter(tokenise(row["text"])) for row in rows]
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document.keys())
    total = len(documents)

    def weight(counter: Counter[str]) -> dict[str, float]:
        return {
            term: frequency * (math.log((1 + total) / (1 + document_frequency[term])) + 1)
            for term, frequency in counter.items()
        }

    query_vector = weight(Counter(tokenise(query)))
    query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
    scored: list[dict] = []
    for row, document in zip(rows, documents):
        vector = weight(document)
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        dot = sum(query_vector.get(term, 0.0) * value for term, value in vector.items())
        scored.append({**row, "retrieval_score": dot / (query_norm * norm)})
    return sorted(scored, key=lambda row: row["retrieval_score"], reverse=True)[:top_k]


"""
Ranks chunks against a query using BM25 (keyword overlap), then re-weights
the raw score using document metadata so active/official policy beats
superseded or draft content, even if the draft shares more words with the
query.
"""

from __future__ import annotations
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import re

from .ingest import Chunk

# How much to trust a chunk based on its document's status/authority.
# Not near-zero on purpose -- we want to suppress, not erase, so the model
# can still SEE a superseded/draft doc if a customer references it directly
# (needed for the prompt-injection test case later).
_AUTHORITY_WEIGHT = {
    ("active", "official"): 1.0,
    ("superseded", "official"): 0.4,
    ("draft", "none"): 0.15,
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class RetrievedChunk:
    chunk: Chunk
    raw_score: float
    weighted_score: float


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._corpus_tokens = [_tokenize(f"{c.heading} {c.text}") for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        q_tokens = _tokenize(query)
        raw_scores = self._bm25.get_scores(q_tokens)
        results = []
        for chunk, raw in zip(self.chunks, raw_scores):
            if raw <= 0:
                continue
            weight = _AUTHORITY_WEIGHT.get((chunk.status, chunk.policy_authority), 0.3)
            results.append(RetrievedChunk(chunk=chunk, raw_score=raw, weighted_score=raw * weight))
        results.sort(key=lambda r: r.weighted_score, reverse=True)
        return results[:top_k]
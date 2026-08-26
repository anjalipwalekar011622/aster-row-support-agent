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


# A small, standard set of English "glue" words that carry no topical
# meaning on their own. Without filtering these, BM25's rarity-based
# scoring can mistake an incidental, meaningless match (e.g. the word
# "long" happening to appear in an unrelated section) for a strong signal,
# drowning out the one keyword that actually matters ("canada"). This is
# especially visible on a small corpus like ours, where a common word can
# accidentally look "rare" just by chance.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "will", "would", "should", "can", "could", "may", "might", "must",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
    "her", "its", "our", "their", "how", "when", "where", "why",
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


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
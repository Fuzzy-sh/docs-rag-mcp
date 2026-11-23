"""Six retrieval methods, all returning ranked SECTIONS (parent) found via CHUNKS (child).

  vector          embeddings only (LangChain book ch07)
  bm25            keyword search only
  hybrid          vector + BM25 merged with Reciprocal Rank Fusion (book ch10, with the
                  de-duplication bug fixed: results are fused by section id, not by position)
  hybrid_rerank   hybrid candidates re-scored by a local cross-encoder (reads query + text together)
  multi_query     Claude writes 3 rephrasings; hybrid for each; fused with RRF (book ch09)
  hyde            Claude writes a hypothetical answer; search with it + the question (book ch09)
"""

import re
from functools import lru_cache

import numpy as np

from .config import RERANK_MODEL, llm
from .index import Index, embed, tokenize

METHODS = ["vector", "bm25", "hybrid", "hybrid_rerank", "multi_query", "hyde"]
POOL = 60      # chunks taken from each list before fusing


def _top_chunks(scores: np.ndarray, n: int = POOL) -> list[int]:
    idx = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
    return [int(i) for i in idx[np.argsort(-scores[idx])] if scores[i] > 0]


def vector_chunks(query: str, n: int = POOL) -> list[int]:
    ix = Index.load()
    return _top_chunks(ix.emb @ embed([query])[0], n)


def bm25_chunks(query: str, n: int = POOL) -> list[int]:
    return _top_chunks(Index.load().bm25.scores(tokenize(query)), n)


def to_sections(chunk_ids: list[int]) -> list[str]:
    """Child -> parent: the rank of a section is the rank of its best chunk."""
    ix, seen, out = Index.load(), set(), []
    for c in chunk_ids:
        sid = ix.chunks[c]["section_id"]
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def rrf(*ranked: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion keyed by the item itself, so the same section found by several
    lists is merged and rises to the top."""
    score: dict[str, float] = {}
    for lst in ranked:
        for rank, item in enumerate(lst):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


def _hybrid(query: str) -> list[str]:
    return rrf(to_sections(vector_chunks(query)), to_sections(bm25_chunks(query)))


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, section_ids: list[str], top: int = 20) -> list[str]:
    ix = Index.load()
    cands = section_ids[:top]
    pairs, owner = [], []
    for sid in cands:
        for c in ix.chunks_of[sid][:4]:                  # score up to 4 chunks per section
            pairs.append((query, ix.chunks[c]["text"][:1500]))
            owner.append(sid)
    scores = _reranker().predict(pairs, show_progress_bar=False)
    best: dict[str, float] = {}
    for sid, s in zip(owner, scores):
        best[sid] = max(best.get(sid, -1e9), float(s))
    return sorted(cands, key=best.get, reverse=True) + section_ids[top:]


def _rephrasings(query: str) -> list[str]:
    out = llm("Rewrite this question about Python, SQL or Git in 3 different ways a reader of "
              "a programming manual might phrase it. Use different wording and likely technical "
              f"terms. One per line, no numbering.\n\nQuestion: {query}", max_tokens=400)
    return [re.sub(r"^[\-\d\.\)\s]+", "", l).strip() for l in out.splitlines() if l.strip()][:3]


def _hypothetical_answer(query: str) -> str:
    return llm("Write a short passage (3-4 sentences) from a programming manual that would answer "
               f"this question. Include likely technical terms or code.\n\nQuestion: {query}",
               max_tokens=500)


def search(query: str, method: str = "vector", k: int = 5) -> list[str]:
    """Return the top-k section ids."""
    if method == "vector":
        ranked = to_sections(vector_chunks(query))
    elif method == "bm25":
        ranked = to_sections(bm25_chunks(query))
    elif method == "hybrid":
        ranked = _hybrid(query)
    elif method == "hybrid_rerank":
        ranked = rerank(query, _hybrid(query))
    elif method == "multi_query":
        ranked = rrf(*[_hybrid(q) for q in [query] + _rephrasings(query)])
    elif method == "hyde":
        passage = _hypothetical_answer(query)
        ranked = rrf(to_sections(vector_chunks(passage)), to_sections(bm25_chunks(query)),
                     to_sections(vector_chunks(query)))
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: {', '.join(METHODS)}")
    return ranked[:k]

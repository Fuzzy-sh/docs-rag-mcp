"""Build and load the search index: chunk embeddings (vector search) + BM25 (keyword search).

Both are kept in plain numpy / Python so every step is visible (no vector DB needed at this size).
"""

import json
import math
import re
from collections import Counter
from collections.abc import Callable
from functools import lru_cache

import numpy as np

from .config import BOOKS, EMBED_MODEL, INDEX_DIR
from .ingest import chunk_section, load_raw, split_sections

STOP = set("a an the and or of to in on for is are be it this that with as by at from how do does "
           "what which can i you your we use using when why".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in STOP and len(t) > 1]


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    return embedder().encode(texts, batch_size=64, normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def build(progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Sections + chunks from data/raw, embed every chunk, save to data/index."""
    sections, chunks = [], []
    for key in BOOKS:
        for page in load_raw(key):
            for sec in split_sections(key, page):
                sections.append(sec)
                for text in chunk_section(sec):
                    chunks.append({"chunk_id": len(chunks), "section_id": sec["id"], "text": text})
    vecs = []
    step = 256
    for i in range(0, len(chunks), step):
        vecs.append(embed([c["text"] for c in chunks[i:i + step]]))
        if progress:
            progress(min(i + step, len(chunks)), len(chunks), "embedding chunks")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", np.vstack(vecs))
    with open(INDEX_DIR / "sections.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in sections)
    with open(INDEX_DIR / "chunks.jsonl", "w") as f:
        f.writelines(json.dumps(c) + "\n" for c in chunks)
    Index.load.cache_clear()
    return {"sections": len(sections), "chunks": len(chunks)}


class BM25:
    """Classic Okapi BM25 over chunk texts."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean())
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


class Index:
    def __init__(self, sections: list[dict], chunks: list[dict], emb: np.ndarray):
        self.sections = {s["id"]: s for s in sections}
        self.chunks = chunks
        self.emb = emb
        self.bm25 = BM25([tokenize(c["text"]) for c in chunks])
        self.chunks_of: dict[str, list[int]] = {}
        for c in chunks:
            self.chunks_of.setdefault(c["section_id"], []).append(c["chunk_id"])

    @staticmethod
    @lru_cache(maxsize=1)
    def load() -> "Index":
        if not (INDEX_DIR / "embeddings.npy").exists():
            raise FileNotFoundError("No index yet: run  python build_index.py")
        sections = [json.loads(l) for l in open(INDEX_DIR / "sections.jsonl")]
        chunks = [json.loads(l) for l in open(INDEX_DIR / "chunks.jsonl")]
        return Index(sections, chunks, np.load(INDEX_DIR / "embeddings.npy"))

"""The same retrieval job done with two frameworks' standard pipelines, for comparison.

  llamaindex      LlamaIndex defaults: whole pages -> SentenceSplitter (1024 tokens, 200 overlap)
                  -> VectorStoreIndex -> retriever
  llamaindex_256  the same, with chunk_size=256 to match the embedding model, which only reads the
                  first 256 tokens of a text (with the default, 84% of chunks were mostly invisible)
  langchain   LangChain's usual tutorial setup: whole pages -> RecursiveCharacterTextSplitter
              (1000 characters, 200 overlap) -> InMemoryVectorStore -> retriever

Both use the SAME local embedding model as the hand-built methods (all-MiniLM-L6-v2), so the
difference measured is each framework's default chunking and retrieval, not the embeddings.

Their chunks don't follow headings, so a retrieved chunk is mapped back to the section(s) of
the page it covers (by word overlap). That keeps the evaluation identical: ranked section ids.
"""

import re
from functools import lru_cache

from .config import BOOKS, EMBED_MODEL, INDEX_DIR
from .index import Index
from .ingest import load_raw, page_slug

LI_DIRS = {1024: INDEX_DIR / "llamaindex", 256: INDEX_DIR / "llamaindex_256"}
LC_PATH = INDEX_DIR / "langchain_store.json"


# ── shared: map a framework chunk back to our sections ───────────────────────
def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


@lru_cache(maxsize=1)
def _page_sections() -> dict[tuple[str, str], list[tuple[str, set]]]:
    """(book, page_slug) -> [(section_id, word 5-grams of its text)]."""
    out: dict[tuple[str, str], list[tuple[str, set]]] = {}
    for s in Index.load().sections.values():
        out.setdefault((s["book"], s["page_slug"]), []).append((s["id"], _shingles(s["text"])))
    return out


def chunk_to_sections(text: str, book: str, title: str) -> list[str]:
    """Sections of this page that the chunk covers, most-covered first."""
    secs = _page_sections().get((book, page_slug(title, BOOKS[book])), [])
    sh = _shingles(text)
    scored = [(len(sh & s) / max(1, len(s)), sid) for sid, s in secs]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(reverse=True)
    covered = [sid for r, sid in scored if r >= 0.5]
    return covered or [sid for _, sid in scored[:1]]


def _rank_sections(chunk_hits: list[tuple[str, str, str]]) -> list[str]:
    """[(text, book, title)] in rank order -> unique section ids in rank order."""
    seen, ranked = set(), []
    for text, book, title in chunk_hits:
        for sid in chunk_to_sections(text, book, title):
            if sid not in seen:
                seen.add(sid)
                ranked.append(sid)
    return ranked


def _pages() -> list[tuple[str, dict]]:
    return [(key, p) for key in BOOKS for p in load_raw(key)]


# ── LlamaIndex ───────────────────────────────────────────────────────────────
@lru_cache(maxsize=2)
def llamaindex_index(chunk_size: int = 1024):
    """Build (or load) a LlamaIndex VectorStoreIndex. 1024 tokens is LlamaIndex's default chunk size;
    256 matches all-MiniLM-L6-v2, which reads only the first 256 tokens of each chunk."""
    from llama_index.core import (Document, Settings, StorageContext, VectorStoreIndex,
                                  load_index_from_storage)
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)   # instead of the OpenAI default
    Settings.llm = None                                                   # retrieval only; Claude answers later
    persist = LI_DIRS[chunk_size]
    if (persist / "docstore.json").exists():
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=str(persist)))
    docs = [Document(text=p["text"], metadata={"book": key, "title": p["title"]}) for key, p in _pages()]
    splitter = SentenceSplitter() if chunk_size == 1024 else SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
    index = VectorStoreIndex.from_documents(docs, transformations=[splitter])
    index.storage_context.persist(persist_dir=str(persist))
    return index


def llamaindex_search(query: str, n: int = 20, chunk_size: int = 1024) -> list[str]:
    nodes = llamaindex_index(chunk_size).as_retriever(similarity_top_k=n).retrieve(query)
    return _rank_sections([(nw.node.get_content(), nw.node.metadata["book"], nw.node.metadata["title"])
                           for nw in nodes])


# ── LangChain ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def langchain_store():
    """Build (or load) a LangChain InMemoryVectorStore with RecursiveCharacterTextSplitter chunks."""
    from langchain_core.documents import Document
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if LC_PATH.exists():
        return InMemoryVectorStore.load(str(LC_PATH), embeddings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(
        [Document(page_content=p["text"], metadata={"book": key, "title": p["title"]}) for key, p in _pages()])
    store = InMemoryVectorStore.from_documents(docs, embeddings)
    store.dump(str(LC_PATH))
    return store


def langchain_search(query: str, n: int = 20) -> list[str]:
    docs = langchain_store().as_retriever(search_kwargs={"k": n}).invoke(query)
    return _rank_sections([(d.page_content, d.metadata["book"], d.metadata["title"]) for d in docs])

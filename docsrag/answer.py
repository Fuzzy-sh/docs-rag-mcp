"""Answer a question from the manuals, citing sections, or say it isn't in the docs."""

import re

from .config import llm
from .index import Index
from .retrieve import search

SYSTEM = """You answer questions using ONLY the numbered documentation sections provided.
- Cite every fact with the section number in brackets, e.g. [2].
- If the sections do not contain the answer, reply starting with exactly "Not in docs:" and say
  briefly what is missing. Do not answer from general knowledge.
- Be concise. Include a short code example if the sections have one."""


def format_sources(section_ids: list[str], max_chars: int = 1800) -> str:
    ix = Index.load()
    parts = []
    for n, sid in enumerate(section_ids, 1):
        s = ix.sections[sid]
        parts.append(f"[{n}] {s['book_title']} > {s['page']} > {s['heading']}\n{s['text'][:max_chars]}")
    return "\n\n".join(parts)


def answer(question: str, method: str = "vector", k: int = 5) -> dict:
    hits = search(question, method, k)
    reply = llm(f"Documentation sections:\n\n{format_sources(hits)}\n\nQuestion: {question}",
                system=SYSTEM, max_tokens=1500)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", reply) if 0 < int(n) <= len(hits)})
    return {"question": question, "method": method, "answer": reply,
            "not_in_docs": reply.lower().startswith("not in docs"),
            "sources": hits, "cited": [hits[n - 1] for n in cited]}

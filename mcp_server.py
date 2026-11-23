"""MCP server for the documentation: the same design as a company-docs chatbot.

Uses each MCP server feature from the book:
  tools       search_docs, get_section, refresh_book (with progress notifications)
  resources   doc://books (list) and doc://{book}/{page} (a whole page, for citations)
  prompt      answer_with_citations
  completion  autocomplete book and page names for the resource template
  errors      ToolError, so the model reads what went wrong and can fix its call

  npx @modelcontextprotocol/inspector python mcp_server.py
  claude mcp add docs-rag -- /path/to/python /full/path/to/mcp_server.py
"""

import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.mcpserver import Context, MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp_types import (Completion, CompletionArgument, CompletionContext,  # noqa: E402
                       PromptReference, ResourceTemplateReference)

from docsrag.config import BOOKS  # noqa: E402
from docsrag.index import Index, build  # noqa: E402
from docsrag.ingest import fetch_book  # noqa: E402
from docsrag.retrieve import METHODS, search  # noqa: E402

mcp = MCPServer("docs-rag")
Method = Literal["vector", "bm25", "hybrid", "hybrid_rerank", "multi_query", "hyde", "llamaindex", "llamaindex_256", "langchain"]


def _section_line(sid: str) -> str:
    s = Index.load().sections[sid]
    preview = " ".join(s["text"].split())[:220]
    return f"- {sid}\n  {s['book_title']} > {s['page']} > {s['heading']}\n  {preview}..."


@mcp.tool()
def search_docs(query: str, method: Method = "vector", k: int = 5) -> str:
    """Search the Python, SQL and Git manuals. Returns section ids with a short preview.
    Then call get_section on the most relevant ids to read them in full before answering.

    Args:
        query: what the user wants to know, in plain words
        method: retrieval method. vector (default) gave the best answers in the evaluation;
            multi_query ranks best but costs one extra Claude call
        k: how many sections to return (1-10)
    """
    if not 1 <= k <= 10:
        raise ToolError("k must be between 1 and 10.")
    hits = search(query, method, k)
    if not hits:
        return "No matching sections."
    return "\n".join(_section_line(s) for s in hits)


@mcp.tool()
def get_section(section_id: str) -> str:
    """Full text of one section, by the id returned from search_docs. Cite it by this id."""
    ix = Index.load()
    s = ix.sections.get(section_id)
    if s is None:
        close = [sid for sid in ix.sections if section_id.split("#")[0] in sid][:5]
        raise ToolError(f"Unknown section id '{section_id}'. Use an id from search_docs."
                        + (f" Similar ids: {', '.join(close)}" if close else ""))
    return f"{s['book_title']} > {s['page']} > {s['heading']}\n(id: {s['id']})\n\n{s['text']}"


@mcp.tool()
async def refresh_book(book: Literal["python", "sql", "git"], ctx: Context) -> str:
    """Re-download one manual from Wikibooks and rebuild the index. Reports progress (slow: ~1 min)."""
    async def report(done: int, total: int, msg: str) -> None:
        await ctx.report_progress(progress=done, total=total, message=msg)

    import asyncio
    loop = asyncio.get_running_loop()

    def progress(done, total, msg):          # called from the worker thread
        asyncio.run_coroutine_threadsafe(report(done, total, msg), loop)

    pages = await asyncio.to_thread(fetch_book, book, progress)
    stats = await asyncio.to_thread(build, progress)
    return f"Refreshed {book}: {len(pages)} pages. Index now has {stats['sections']} sections."


@mcp.resource("doc://books")
def list_books() -> str:
    """The manuals in this server, with their page counts."""
    ix = Index.load()
    lines = []
    for key, title in BOOKS.items():
        pages = sorted({s["page_slug"] for s in ix.sections.values() if s["book"] == key})
        lines.append(f"{key}: {title.replace('_', ' ')} ({len(pages)} pages)")
    return "\n".join(lines)


@mcp.resource("doc://{book}/{page}")
def read_page(book: str, page: str) -> str:
    """Every section of one page, e.g. doc://python/Lists."""
    secs = [s for s in Index.load().sections.values() if s["book"] == book and s["page_slug"] == page]
    if not secs:
        raise ToolError(f"No page '{page}' in book '{book}'. Read doc://books for the books.")
    return "\n\n".join(f"## {s['heading']}  (id: {s['id']})\n{s['text']}" for s in secs)


@mcp.prompt()
def answer_with_citations(question: str) -> str:
    """Answer a question from the manuals, citing section ids, or say it isn't documented."""
    return (f"Answer this question using only the documentation server: {question}\n\n"
            "1. Call search_docs with the question.\n"
            "2. Call get_section for the 2-3 most relevant ids and read them.\n"
            "3. Answer concisely and cite each fact with its section id in brackets.\n"
            "4. If the sections don't contain the answer, say 'Not in docs' and what is missing.")


@mcp.completion()
async def complete(ref: PromptReference | ResourceTemplateReference, argument: CompletionArgument,
                   context: CompletionContext | None) -> Completion:
    """Autocomplete book keys and page names for doc://{book}/{page}."""
    values: list[str] = []
    if isinstance(ref, ResourceTemplateReference) and ref.uri == "doc://{book}/{page}":
        typed = argument.value.lower()
        if argument.name == "book":
            values = [k for k in BOOKS if k.startswith(typed)]
        elif argument.name == "page":
            book = (context.arguments or {}).get("book") if context else None
            pages = {s["page_slug"] for s in Index.load().sections.values() if not book or s["book"] == book}
            values = sorted(p for p in pages if typed in p.lower())[:50]
    return Completion(values=values, total=len(values), has_more=False)


if __name__ == "__main__":
    mcp.run()

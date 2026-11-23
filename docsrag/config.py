"""Paths, models, API key, and a tiny Claude helper that counts calls and tokens."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"

# Wikibooks manuals used as the "documentation" (CC BY-SA 4.0, see README for attribution).
BOOKS = {
    "python": "Python_Programming",
    "sql": "Structured_Query_Language",
    "git": "Git",
}
WIKI_API = "https://en.wikibooks.org/w/api.php"
USER_AGENT = "docs-rag-mcp/0.1 (educational RAG demo)"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"     # local, free (Claude has no embedding model)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"      # local cross-encoder re-ranker

load_dotenv(ROOT / ".env", override=True) or load_dotenv(find_dotenv(usecwd=True), override=True)
os.environ["ANTHROPIC_API_KEY"] = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")

LLM_MODEL = os.environ.get("DOCS_MODEL", "claude-haiku-5-5")
JUDGE_MODEL = os.environ.get("DOCS_JUDGE_MODEL", "claude-sonnet-5-5")

USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "truncated": 0}
_NO_THINKING: dict[str, str] = {}   # model -> the "thinking off" setting it accepts


def llm(prompt: str, system: str | None = None, model: str | None = None, max_tokens: int = 1500) -> str:
    """One Claude call; returns the text. Usage is added to USAGE so the eval can report cost.

    Extended thinking is switched off: these are short tasks, and thinking tokens count against
    max_tokens, which silently cut replies short (or left them empty) in the first evaluation run.
    Replies that still hit the limit are counted in USAGE["truncated"]."""
    import anthropic

    if not os.environ["ANTHROPIC_API_KEY"]:
        raise RuntimeError("Add LLM_API_KEY=sk-ant-... to a .env file (see .env.example).")
    client = anthropic.Anthropic()
    model = model or LLM_MODEL
    kwargs = {"system": system} if system else {}
    # Models differ in how thinking is turned off: most accept {"type": "disabled"}; some
    # (e.g. claude-sonnet-5-5) require {"type": "between_tools"}. Learn it once per model.
    mode = _NO_THINKING.get(model, "disabled")
    try:
        msg = client.messages.create(model=model, max_tokens=max_tokens, thinking={"type": mode},
                                     messages=[{"role": "user", "content": prompt}], **kwargs)
    except anthropic.BadRequestError as e:
        if mode != "disabled" or "between_tools" not in str(e):
            raise
        _NO_THINKING[model] = "between_tools"
        msg = client.messages.create(model=model, max_tokens=max_tokens, thinking={"type": "between_tools"},
                                     messages=[{"role": "user", "content": prompt}], **kwargs)
    USAGE["truncated"] += msg.stop_reason == "max_tokens"
    USAGE["calls"] += 1
    USAGE["input_tokens"] += msg.usage.input_tokens
    USAGE["output_tokens"] += msg.usage.output_tokens
    return "".join(b.text for b in msg.content if b.type == "text").strip()

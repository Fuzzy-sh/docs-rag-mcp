"""Download the Wikibooks manuals and split them into sections and chunks.

Section = one heading in a page (the unit we cite and evaluate).
Chunk   = what we embed and search: a section, or a piece of a long section, prefixed with
          its location ("Python Programming > Lists > Overview") so search sees the context.
          A chunk always points back to its section (parent/child retrieval).
"""

import json
import re
import time
from collections.abc import Callable

import requests

from .config import BOOKS, RAW_DIR, USER_AGENT, WIKI_API

SKIP_TITLES = re.compile(r"print[ _]version|all[ _]chapters|printable|/contents$|/index$|/authors$|"
                         r"/print$|/single[ _]page", re.I)
SKIP_HEADINGS = {"references", "external links", "see also", "further reading", "notes", "sources"}
HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")


def _get(params: dict) -> dict:
    """GET with polite retries: honours Retry-After on 429 (rate limit), backs off otherwise."""
    for attempt in range(7):
        r = requests.get(WIKI_API, params={**params, "format": "json", "formatversion": 2,
                                           "maxlag": 5},
                         headers={"User-Agent": USER_AGENT}, timeout=30)
        if r.status_code == 200 and "error" not in r.json():
            return r.json()
        wait = float(r.headers.get("Retry-After") or 0) or min(60, 3 * 2 ** attempt)
        time.sleep(wait)
    r.raise_for_status()
    raise RuntimeError(f"Wikibooks API error: {r.text[:200]}")


def list_pages(book_title: str) -> list[str]:
    data = _get({"action": "query", "list": "allpages", "apprefix": f"{book_title}/",
                 "apnamespace": 0, "aplimit": 500})
    titles = [book_title] + [p["title"] for p in data["query"]["allpages"]]
    return [t for t in titles if not SKIP_TITLES.search(t)]


def fetch_page(title: str) -> dict:
    data = _get({"action": "query", "prop": "extracts|info", "explaintext": 1,
                 "exsectionformat": "wiki", "titles": title})
    page = data["query"]["pages"][0]
    return {"title": page["title"], "revid": page.get("lastrevid"), "text": page.get("extract", "")}


def fetch_book(key: str, progress: Callable[[int, int, str], None] | None = None) -> list[dict]:
    """Download every page of one book to data/raw/<key>.jsonl."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    titles = list_pages(BOOKS[key])
    partial = RAW_DIR / f"{key}.partial.jsonl"           # resume if a previous run was interrupted
    done = {json.loads(l)["title"]: json.loads(l) for l in open(partial)} if partial.exists() else {}
    with open(partial, "a") as f:
        for i, t in enumerate(titles, 1):
            if t.replace("_", " ") not in done:
                p = fetch_page(t)
                done[p["title"]] = p
                f.write(json.dumps(p) + "\n")
                f.flush()
                time.sleep(0.6)  # be polite to the API (about 100 pages a minute)
            if progress:
                progress(i, len(titles), f"{key}: {t}")
    pages = [p for p in done.values() if len(p["text"]) > 200]
    with open(RAW_DIR / f"{key}.jsonl", "w") as f:
        for p in pages:
            f.write(json.dumps(p) + "\n")
    partial.unlink()
    return pages


def load_raw(key: str) -> list[dict]:
    return [json.loads(line) for line in open(RAW_DIR / f"{key}.jsonl")]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "intro"


def page_slug(title: str, book_title: str) -> str:
    # The API returns "Python Programming/Lists" (spaces); BOOKS uses "Python_Programming".
    title, book_title = title.replace("_", " "), book_title.replace("_", " ")
    short = title[len(book_title):].lstrip("/") if title.startswith(book_title) else title
    return (short or "Home").replace(" ", "_").replace("/", "~")


def split_sections(key: str, page: dict) -> list[dict]:
    book_title = BOOKS[key]
    pslug = page_slug(page["title"], book_title)
    page_name = pslug.replace("_", " ").replace("~", " / ")
    sections, path, buf = [], [], []

    def flush():
        text = "\n".join(buf).strip()
        buf.clear()
        heading = " > ".join(path) if path else "Introduction"
        if len(text) < 80 or (path and path[-1].lower() in SKIP_HEADINGS):
            return
        sid = f"{key}/{pslug}#{_slug(heading)}"
        if any(s["id"] == sid for s in sections):          # duplicate heading names in one page
            sid = f"{sid}-{len(sections)}"
        sections.append({"id": sid, "book": key, "book_title": book_title.replace("_", " "),
                         "page": page_name, "page_slug": pslug, "heading": heading,
                         "revid": page["revid"], "text": text})

    for line in page["text"].splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1)) - 2          # "==" is level 0
            path = path[:level] + [m.group(2)]
        else:
            buf.append(line)
    flush()
    return sections


def _overlap_tail(text: str, overlap: int) -> str:
    """The last ~overlap characters, starting at a sentence or line boundary (not mid-word)."""
    tail = text[-overlap:]
    for sep in ("\n", ". ", "; ", " "):
        i = tail.find(sep)
        if 0 <= i < len(tail) - 20:
            return tail[i + len(sep):]
    return tail


def chunk_section(sec: dict, size: int = 900, overlap: int = 150) -> list[str]:
    """Split a long section on paragraph boundaries into ~size-character chunks with overlap.
    Each chunk is prefixed with its location: a 'contextual chunk header'."""
    header = f"{sec['book_title']} > {sec['page']} > {sec['heading']}\n"
    text = sec["text"]
    if len(text) <= size * 1.3:
        return [header + text]
    paras, chunks, cur = [p for p in re.split(r"\n\s*\n", text) if p.strip()], [], ""
    for p in paras:
        while len(p) > size * 1.5:                      # a single huge paragraph (e.g. code)
            chunks.append(header + (cur + "\n\n" + p[:size]).strip())
            cur, p = "", p[size - overlap:]
        if cur and len(cur) + len(p) > size:
            chunks.append(header + cur.strip())
            cur = _overlap_tail(cur, overlap)           # carry a little context forward
        cur += "\n\n" + p
    if cur.strip():
        chunks.append(header + cur.strip())
    return chunks

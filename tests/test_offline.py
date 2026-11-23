"""Offline tests (no API key, no download): splitting, chunking, BM25 and RRF.

  python tests/test_offline.py      or      pytest tests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docsrag.index import BM25, tokenize  # noqa: E402
from docsrag.ingest import chunk_section, split_sections  # noqa: E402
from docsrag.retrieve import rrf  # noqa: E402

PAGE = {"title": "Git/Undoing", "revid": 1, "text": (
    "Intro text that is long enough to keep as its own section, more than eighty characters here.\n"
    "== Reset ==\nUse git reset --soft HEAD~1 to undo the last commit but keep the changes staged. "
    "This is the most common way.\n"
    "=== Hard reset ===\ngit reset --hard throws the changes away completely; be careful with it, "
    "it cannot be undone easily.\n"
    "== See also ==\nLinks to other pages that should be dropped because it is only links here."
)}


def test_split_by_headings():
    secs = split_sections("git", PAGE)
    heads = [s["heading"] for s in secs]
    assert heads == ["Introduction", "Reset", "Reset > Hard reset"], heads   # 'See also' dropped
    assert secs[2]["id"] == "git/Undoing#reset-hard-reset"


def test_chunks_have_context_header_and_overlap():
    sec = {"book_title": "Git", "page": "Undoing", "heading": "Reset",
           "text": "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(8))}
    chunks = chunk_section(sec, size=900, overlap=150)
    assert len(chunks) > 1
    assert all(c.startswith("Git > Undoing > Reset\n") for c in chunks)


def test_bm25_prefers_matching_doc():
    docs = [tokenize("git reset undo the last commit"), tokenize("python list append extend")]
    s = BM25(docs).scores(tokenize("undo commit"))
    assert s[0] > s[1] == 0


def test_rrf_merges_same_item():
    # 'b' is 2nd in both lists, so it must beat 'a' and 'c', each 1st in only one list
    fused = rrf(["a", "b", "x"], ["c", "b", "y"])
    assert fused[0] == "b", fused


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)

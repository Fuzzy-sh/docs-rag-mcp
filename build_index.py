"""Download the three Wikibooks manuals and build the search index.

  python build_index.py            # download + index
  python build_index.py --no-fetch # re-index the already-downloaded pages
"""

import argparse
import json
import sys

from docsrag.config import BOOKS, ROOT
from docsrag.index import build
from docsrag.ingest import fetch_book, load_raw


def bar(done: int, total: int, msg: str) -> None:
    sys.stdout.write(f"\r  {done}/{total}  {msg[:60]:<60}")
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    if not a.no_fetch:
        for key in BOOKS:
            pages = fetch_book(key, bar)
            print(f"\n{key}: {len(pages)} pages")
    snapshot = {k: {p["title"]: p["revid"] for p in load_raw(k)} for k in BOOKS}
    (ROOT / "data" / "snapshot_revisions.json").write_text(json.dumps(snapshot, indent=1))
    stats = build(bar)
    print(f"\nIndexed {stats['sections']} sections in {stats['chunks']} chunks.")


if __name__ == "__main__":
    main()

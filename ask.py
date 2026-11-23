"""Ask the manuals a question from the terminal.

  python ask.py "How do I undo my last commit?"
  python ask.py --method bm25 --search-only "list comprehension"
"""

import argparse

from docsrag.answer import answer
from docsrag.index import Index
from docsrag.retrieve import METHODS, search


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--method", choices=METHODS, default="vector")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--search-only", action="store_true")
    a = p.parse_args()

    ix = Index.load()
    if a.search_only:
        for n, sid in enumerate(search(a.question, a.method, a.k), 1):
            s = ix.sections[sid]
            print(f"{n}. {s['book_title']} > {s['page']} > {s['heading']}   ({sid})")
        return
    r = answer(a.question, a.method, a.k)
    print(r["answer"], "\n\nSources:")
    for n, sid in enumerate(r["sources"], 1):
        s = ix.sections[sid]
        mark = "*" if sid in r["cited"] else " "
        print(f" {mark}[{n}] {s['book_title']} > {s['page']} > {s['heading']}")


if __name__ == "__main__":
    main()

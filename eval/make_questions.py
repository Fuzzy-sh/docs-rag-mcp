"""Create the labelled test set: eval/questions.jsonl

For a fixed random sample of sections (same seed = same sample), Claude writes the question a
real reader would ask whose answer is in that section, in their OWN words (not copying the
heading). That section becomes the gold label. Plus hand-written out-of-scope questions, which
the chatbot must answer with "Not in docs".

Generated questions are a known shortcut: they are spot-checked by hand (see README).
"""

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docsrag.config import JUDGE_MODEL, llm  # noqa: E402
from docsrag.index import Index  # noqa: E402

PER_BOOK = {"python": 18, "sql": 14, "git": 10}
OUT_OF_SCOPE = [
    "How do I configure a VLAN on a Cisco switch?",
    "What is a good recipe for sourdough bread?",
    "How do I set up autoscaling for a Kubernetes deployment?",
    "What is the difference between useEffect and useLayoutEffect in React?",
    "How do I file a personal tax return in Canada?",
    "How do I change the oil in a 2015 Toyota Corolla?",
]

PROMPT = """Here is a section from a programming manual ({book}).

Location: {where}
---
{text}
---

Write ONE question that a developer might type into a documentation chatbot, which THIS section
answers. Rules:
- Use your own natural words, like a real user; do not copy the heading or unusual phrases.
- It must be specific enough that this section is a good answer (not a vague "what is Python").
- If the section is only links, a table of contents, or too thin to answer anything, return skip.

Reply with JSON only: {{"question": "...", "answer": "one-sentence answer from the section"}}
or {{"skip": true}}"""


def main() -> None:
    ix = Index.load()
    rng = random.Random(7)
    out = []
    for book, n in PER_BOOK.items():
        pool = [s for s in ix.sections.values() if s["book"] == book and 300 <= len(s["text"]) <= 4000]
        rng.shuffle(pool)
        kept = 0
        for s in pool:
            if kept == n:
                break
            reply = llm(PROMPT.format(book=s["book_title"], where=f"{s['page']} > {s['heading']}",
                                      text=s["text"][:3000]), model=JUDGE_MODEL, max_tokens=600)
            m = re.search(r"\{.*\}", reply, re.S)
            data = json.loads(m.group(0)) if m else {"skip": True}
            if data.get("skip") or not data.get("question"):
                continue
            kept += 1
            out.append({"id": f"{book}-{kept:02d}", "book": book, "question": data["question"],
                        "reference_answer": data.get("answer", ""), "gold_section": s["id"],
                        "gold_page": s["id"].split("#")[0], "in_scope": True})
            print(f"{out[-1]['id']}: {data['question']}\n      <- {s['page']} > {s['heading']}")
    for i, q in enumerate(OUT_OF_SCOPE, 1):
        out.append({"id": f"out-{i:02d}", "book": None, "question": q, "reference_answer": "",
                    "gold_section": None, "gold_page": None, "in_scope": False})
    path = ROOT / "eval" / "questions.jsonl"
    path.write_text("".join(json.dumps(q) + "\n" for q in out))
    print(f"\nWrote {len(out)} questions to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Builds notebooks/01_docs_rag_walkthrough.ipynb from the cells below, then runs it.

  python notebooks/make_notebook.py          # build + execute (needs the index and eval results)

Kept as a script so the notebook's text and code stay reviewable in plain Python.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
OUT = HERE / "01_docs_rag_walkthrough.ipynb"

md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

cells = [
md("""# Documentation Q&A over Wikibooks — Walkthrough

A documentation chatbot built the way a company-docs assistant is built: **ingest** manuals,
**index** them two ways, **retrieve** with six methods, **answer with citations** (or say
*"Not in docs"*), expose it all as an **MCP server**, and **measure** every step.

The "documentation" is three Wikibooks manuals: *Python Programming*, *Structured Query Language*
and *Git* (CC BY-SA 4.0). Every output below is real: produced by running this notebook against the
built index and the saved evaluation results.

**Sections:** 1. Data and ingestion · 2. Indexing · 3. Retrieval, six methods · 4. Answering with
citations · 5. The MCP server · 6. Evaluation · 7. Findings

Prerequisites: `python build_index.py` (downloads + indexes, ~20 min the first time) and
`python eval/run_eval.py` (writes `eval/results/`). Live Claude calls in this notebook: about 6."""),

code("""import json, os, subprocess, sys, warnings
from pathlib import Path
from statistics import median

import logging
warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from IPython.display import Markdown, display
from docsrag.config import BOOKS, USAGE
from docsrag.ingest import load_raw, split_sections, chunk_section

def table(rows, headers):
    \"\"\"Render a list of rows as a Markdown table.\"\"\"
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    display(Markdown("\\n".join(out)))"""),

md("""## 1. The data and ingestion

`build_index.py` downloaded every page of the three books through the Wikibooks API as **plain
text with `== headings ==`**, one page per request, waiting whenever the API rate-limited it. Pages
under 200 characters (navigation stubs) and "Print version" pages (which duplicate whole books) were dropped."""),

code("""rows = []
for key in BOOKS:
    pages = load_raw(key)
    rows.append([BOOKS[key].replace("_", " "), len(pages), f"{sum(len(p['text']) for p in pages):,}"])
table(rows, ["Book", "Pages kept", "Characters"])"""),

md("The text is small (well under 1 MB). One real page, *Python Programming / Lists*:"),

code("""page = next(p for p in load_raw("python") if p["title"] == "Python Programming/Lists")
print(f"{page['title']}  |  {len(page['text']):,} characters  |  revision {page['revid']}\\n")
print(page["text"][:500])"""),

md("""### Split at headings → sections

Each `== Heading ==` starts a **section**; sub-headings keep their parent in the path. A section is
the unit the chatbot **cites** and the evaluation **scores**, so it gets a stable id
`book/page#heading-path`. *See also / References / External links* and sections under 80 characters
are dropped.

Why headings and not fixed-size pieces? The authors already split each topic into meaningful
parts; cutting every N characters would cut examples in half."""),

code("""sections = split_sections("python", page)
print(f"{len(sections)} sections\\n")
table([[len(s["text"]), s["heading"], f"`{s['id']}`"] for s in sections[:10]],
      ["Chars", "Heading path", "Section id"])"""),

md("""### Cut long sections → chunks

Sections up to ~1,170 characters stay **one chunk**. Longer ones are cut at blank lines into
~900-character chunks; each new chunk repeats the last ~150 characters of the previous one
(**overlap**, starting at a sentence boundary) so a sentence on a cut is still findable. Every chunk
starts with its location: a **contextual chunk header**, so search knows the chunk is about
*Python lists* even if its own text never says so."""),

code("""sec = next(s for s in sections if s["heading"].endswith("List comprehensions"))
chunks = chunk_section(sec)
print(f"Section '{sec['heading']}': {len(sec['text'])} chars -> {len(chunks)} chunks\\n")
for i, c in enumerate(chunks, 1):
    header, body = c.split("\\n", 1)
    print(f"chunk {i} ({len(c)} chars)\\n  header: {header}\\n  starts: {body[:90]!r}\\n")"""),

code("""rows = []
for key in BOOKS:
    secs = [s for p in load_raw(key) for s in split_sections(key, p)]
    chs = [c for s in secs for c in chunk_section(s)]
    lens = [len(c) for c in chs]
    rows.append([key, len(secs), len(chs), min(lens), int(median(lens)), max(lens)])
table(rows, ["Book", "Sections", "Chunks", "Min chars", "Median chars", "Max chars"])"""),

md("""## 2. Indexing: two ways to search

Every chunk is indexed **twice**:

- **Vector (meaning):** a local model (`all-MiniLM-L6-v2`; Claude has no embedding model) turns each
  chunk into **384 numbers**. Similar meaning → similar numbers, even with different words.
- **BM25 (keywords):** word counts per chunk, weighted by how rare each word is.

No vector database: 1,261 × 384 numbers fit in memory, and one matrix multiply compares a question
with every chunk in milliseconds."""),

code("""from docsrag.index import Index, embed, tokenize
ix = Index.load()
print(f"{len(ix.sections)} sections, {len(ix.chunks)} chunks")
print(f"embeddings: {ix.emb.shape[0]} x {ix.emb.shape[1]} numbers  ({ix.emb.nbytes/1e6:.1f} MB)")
print("first 8 numbers of chunk 0:", ix.emb[0][:8].round(3))"""),

md("Meaning, not words: *undo a commit* is close to *revert the last change* (no shared words) and far from *sort a list*:"),

code("""phrases = ["undo a commit", "revert the last change", "sort a list in Python", "bake sourdough bread"]
v = embed(phrases)
table([[p, f"{float(v[0] @ v[i]):.2f}"] for i, p in enumerate(phrases)], ["Phrase", "Similarity to 'undo a commit'"])"""),

md("BM25 weights rare words higher. For one question, the rarity (`idf`) of each kept word:"),

code("""q = "How do I undo my last commit but keep my changes?"
toks = tokenize(q)
print("tokens kept:", toks)
table([[t, f"{ix.bm25.idf.get(t, 0):.2f}"] for t in toks], ["Word", "Rarity (idf)"])"""),

md("""## 3. Retrieval: six methods

All six return ranked **sections**, found through their chunks (search small, answer big):

| Method | How |
|---|---|
| `vector` | embed the question; cosine similarity with every chunk |
| `bm25` | keyword scoring |
| `hybrid` | vector + BM25 merged with **Reciprocal Rank Fusion** |
| `hybrid_rerank` | hybrid top 20 re-scored by a local **cross-encoder** reading question + chunk together |
| `multi_query` | Claude writes 3 rephrasings; hybrid for each; fused |
| `hyde` | Claude writes a hypothetical answer; search with it, fused with the question's results |

One real question, *"How do I undo my last commit but keep my changes?"*. The section that answers
it is **Git > Introduction > Removing files > Resetting a commit** (`git reset` "cancels the
commit(s) and leaves the file(s) uncommitted")."""),

code("""from docsrag.retrieve import search, METHODS
GOLD = "git/Introduction#removing-files-resetting-a-commit"

def show(query, method, k=5):
    ranked = search(query, method, k=10)
    rank = ranked.index(GOLD) + 1 if GOLD in ranked else "not in top 10"
    print(f"{method:14} gold rank: {rank}")
    for i, sid in enumerate(ranked[:k], 1):
        s = ix.sections[sid]
        print(f"   {i}. {'✓ ' if sid == GOLD else ''}{s['book_title']} > {s['page']} > {s['heading']}")

for m in ["vector", "bm25", "hybrid", "hybrid_rerank"]:
    show(q, m)"""),

md("""**What happened:** vector search understands *undo* ≈ *cancel* and ranks the right section high.
BM25 misses it: the question's words aren't in the text. Fusion rewards sections that **both** lists
found, so the vector-only hit sinks; re-ranking lifts it but not into the top 5. One example proves
nothing, which is why section 6 measures 42 questions.

### Reciprocal Rank Fusion, and the book's bug

RRF scores each item `1 / (60 + rank)` in each list and adds them up. It must be keyed by the
**item**. The LangChain book's chapter 10 version keyed by *position*, so the same document found by
two lists was never merged:"""),

code("""from docsrag.retrieve import rrf
list_a, list_b = ["A", "B", "X"], ["C", "B", "Y"]   # B is 2nd in both lists

def rrf_by_position(*lists, k=60):                  # the book's version: key = (list, position)
    scores = {(i, r): 1/(k + r + 1) for i, l in enumerate(lists) for r, _ in enumerate(l)}
    order = sorted(scores, key=scores.get, reverse=True)
    return [lists[i][r] for i, r in order]

print("by position (bug):", rrf_by_position(list_a, list_b))
print("by item (fixed):  ", rrf(list_a, list_b))"""),

md("With the bug, **B** appears twice and never beats the single hits; fixed, B is merged and ranks first."),

md("### Multi-query and HyDE (live Claude calls)"),

code("""from docsrag.retrieve import _rephrasings, _hypothetical_answer
print("Rephrasings:"); [print("  -", r) for r in _rephrasings(q)]
print("\\nHypothetical answer (HyDE):\\n ", _hypothetical_answer(q)[:400])
print()
for m in ["multi_query", "hyde"]:
    show(q, m)"""),

md("""## 4. Answering with citations

The top 5 sections (default method: `vector`, the best for answers in section 6) go to Claude (Haiku), numbered `[1]`–`[5]`, with rules: use only these sections,
cite every fact, and if the answer isn't there start with **"Not in docs:"**. The same rule as a
company-docs bot: cite or say you don't know."""),

code("""from docsrag.answer import answer

def ask(question, method="vector"):
    r = answer(question, method)
    print(f"Q: {question}\\n\\n{r['answer']}\\n\\nSources:")
    for i, sid in enumerate(r["sources"], 1):
        s = ix.sections[sid]
        print(f"  {'*' if sid in r['cited'] else ' '}[{i}] {s['book_title']} > {s['page']} > {s['heading']}")
    print()

ask("How do I get the elements that are in either of two Python sets but not in both?")
ask("What is a good recipe for sourdough bread?")"""),

md("""## 5. The MCP server

`mcp_server.py` exposes the same pipeline to any MCP client (Claude Code, the Inspector…) and uses
each server feature from *AI Agents with MCP*:

| Feature | Here |
|---|---|
| Tools | `search_docs`, `get_section`, `refresh_book` (with **progress notifications**) |
| Resources | `doc://books`, and the template `doc://{book}/{page}` |
| Prompt | `answer_with_citations` |
| Completion | autocompletes book and page names for the resource template |
| Errors | `ToolError`, so the model reads what went wrong |

Below, a raw stdio session: the same JSON-RPC messages a client sends."""),

code("""msgs = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18",
     "capabilities": {}, "clientInfo": {"name": "notebook", "version": "1"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {"jsonrpc": "2.0", "id": 3, "method": "resources/templates/list"},
    {"jsonrpc": "2.0", "id": 4, "method": "prompts/list"},
    {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "search_docs",
     "arguments": {"query": "insert or update a row in one statement", "k": 3}}},
    {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "get_section",
     "arguments": {"section_id": "sql/MERGE#does-not-exist"}}},
    {"jsonrpc": "2.0", "id": 7, "method": "completion/complete", "params": {
     "ref": {"type": "ref/resource", "uri": "doc://{book}/{page}"},
     "argument": {"name": "page", "value": "lis"}, "context": {"arguments": {"book": "python"}}}},
]
script = "import sys,time\\n" + "".join(f"print({json.dumps(json.dumps(m))}, flush=True)\\n" for m in msgs) + "time.sleep(25)\\n"
feeder = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE)
server = subprocess.Popen([sys.executable, str(ROOT / "mcp_server.py")], stdin=feeder.stdout,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=ROOT)
replies = {}
while len(replies) < 7:
    line = server.stdout.readline()
    if not line:
        break
    d = json.loads(line)
    if "id" in d:
        replies[d["id"]] = d.get("result", d.get("error"))
server.kill(); feeder.kill()

print("tools:     ", [t["name"] for t in replies[2]["tools"]])
print("templates: ", [t["uriTemplate"] for t in replies[3]["resourceTemplates"]])
print("prompts:   ", [p["name"] for p in replies[4]["prompts"]])
print("\\nsearch_docs ->\\n" + replies[5]["content"][0]["text"])
print("\\nget_section with a bad id -> isError =", replies[6].get("isError"))
print("  " + replies[6]["content"][0]["text"][:220])
print("\\ncompletion for doc://python/lis... ->", replies[7]["completion"]["values"][:6])"""),

md("""## 6. Evaluation

### The test set

For a fixed random sample of sections, Claude (Sonnet) wrote the question a reader would ask
**in their own words**; that section is the gold label. Plus 6 hand-written **out-of-scope**
questions the bot must refuse."""),

code("""questions = [json.loads(l) for l in open(ROOT / "eval/questions.jsonl")]
inscope = [x for x in questions if x["in_scope"]]
print(f"{len(questions)} questions: {len(inscope)} in scope "
      f"({sum(x['book']=='python' for x in inscope)} Python, {sum(x['book']=='sql' for x in inscope)} SQL, "
      f"{sum(x['book']=='git' for x in inscope)} Git) + {len(questions)-len(inscope)} out of scope\\n")
table([[x["id"], x["question"][:95], x["gold_section"] or "—"] for x in questions[::7]],
      ["id", "Question", "Gold section"])"""),

md("""### Retrieval: which method finds the right section?

- **hit@1 / hit@5:** the gold section is ranked 1st / in the top 5
- **page@5:** a section of the gold *page* is in the top 5
- **MRR@10:** 1 / rank of the gold section (0 if not in the top 10), averaged"""),

code("""ret = json.load(open(ROOT / "eval/results/retrieval.json"))
S = ret["summary"]
table([[m, f"{s['hit@1']:.2f}", f"{s['hit@5']:.2f}", f"{s['page@5']:.2f}", f"{s['MRR@10']:.2f}",
        f"{s['avg_ms']} ms", s["llm_calls_per_q"]] for m, s in S.items()],
      ["Method", "hit@1", "hit@5", "page@5", "MRR@10", "Avg time", "Claude calls"])"""),

md("Where the methods **disagree** (gold rank per method; `—` = not in the top 10):"),

code("""by = {}
for r in ret["rows"]:
    by.setdefault(r["id"], {})[r["method"]] = r["rank"]
rows = [[qid] + [d.get(m) or "—" for m in METHODS] for qid, d in by.items()
        if len({d.get(m) for m in METHODS}) > 1 and any(d.get(m) != 1 for m in METHODS)]
table(rows, ["id"] + METHODS)"""),

md("""### Answers: correct, grounded, cited, and refusing when it should

A judge model (Sonnet) compares each answer with the reference answer (**correct**) and checks every
claim against the retrieved sections (**grounded**)."""),

code("""ans = json.load(open(ROOT / "eval/results/answers.json"))
A = ans["summary"]
table([[m, f"{s['correct']:.2f}", f"{s['grounded']:.2f}", f"{s['cites_gold']:.2f}",
        s["refusals_out_of_scope"], s["false_refusals"]] for m, s in A.items()],
      ["Retrieval method", "Correct", "Grounded", "Cites gold", "Refused out-of-scope", "False refusals"])"""),

md("Every in-scope miss, for both methods: answered **\"Not in docs\"** (false refusal), judged incorrect, or not grounded:"),

code("""qinfo = {x["id"]: x for x in questions}
for m in A:
    print(f"== {m}")
    for r in ans["rows"]:
        j = r.get("judge", {})
        if r["method"] == m and r["in_scope"] and (r["not_in_docs"] or not j.get("correct") or not j.get("grounded")):
            tags = [t for t, bad in [("NOT IN DOCS", r["not_in_docs"]), ("INCORRECT", not j.get("correct")),
                                     ("NOT GROUNDED", not j.get("grounded"))] if bad]
            found = "gold retrieved" if qinfo[r["id"]]["gold_section"] in r["sources"] else "gold NOT retrieved"
            print(f"  [{', '.join(tags)}] {r['id']} ({found}): {qinfo[r['id']]['question'][:80]}")
            print(f"      judge: {j.get('why', '')[:150]}")
    print()"""),

md("FINDINGS_PLACEHOLDER"),

code("""print(f"Claude calls made by this notebook: {USAGE['calls']} "
      f"({USAGE['input_tokens']:,} input / {USAGE['output_tokens']:,} output tokens, {USAGE['truncated']} cut off)")"""),
]

nb = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})

if __name__ == "__main__":
    import sys
    findings = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else "## 7. Findings\n\n(see README)"
    for c in nb.cells:
        if c.cell_type == "markdown" and c.source == "FINDINGS_PLACEHOLDER":
            c.source = findings
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, OUT)
    print(f"Wrote {OUT.relative_to(HERE.parent)} ({len(nb.cells)} cells)")

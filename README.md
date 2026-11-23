# Docs RAG + MCP: documentation Q&A, nine retrieval setups, measured

A documentation chatbot built the way a company-docs assistant is built: **ingest** manuals,
**index** them two ways, **retrieve** with six hand-built methods (plus LangChain's and LlamaIndex's
standard pipelines for comparison), **answer with citations** (or say *"Not in docs"*), expose it all as
an **MCP server**, and **measure** every step, with every run tracked in **MLflow**.

The "documentation" is three Wikibooks manuals: *Python Programming*, *Structured Query Language*
and *Git* (171 pages → 926 sections → 1,261 chunks).

**📓 Start here:** [`notebooks/01_docs_rag_walkthrough.ipynb`](notebooks/01_docs_rag_walkthrough.ipynb)
walks through the whole pipeline with real outputs: chunking a page, embeddings vs keywords, the six
methods on one question, **how to do the same with LlamaIndex and LangChain**, a cited answer, a raw MCP
session, the evaluation, and **MLflow** run tracking and tracing.
**How it works, in depth:** [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).

## Pipeline

```
Wikibooks API ──▶ sections (by heading) ──▶ chunks (~900 chars, overlap, location header)
                                              │
                         ┌────────────────────┴───────────────────┐
                  vector index (local embeddings)          BM25 keyword index
                         └────────────────────┬───────────────────┘
            vector · bm25 · hybrid (RRF) · hybrid + re-rank · multi-query · HyDE
                                              │
                    top-5 sections ──▶ Claude answers with [citations] or "Not in docs"
                                              │
                      MCP server: tools · resources · prompt · completion · progress
```

## Results (42 in-scope + 6 off-topic questions)

**Retrieval**: does the method rank the section that answers the question?

| Method | hit@1 | hit@5 | MRR@10 | Time | Claude calls |
|---|---|---|---|---|---|
| vector | 0.67 | **1.00** | 0.80 | 161 ms | 0 |
| bm25 | 0.69 | 0.88 | 0.77 | **2 ms** | 0 |
| hybrid (vector + BM25, RRF) | 0.64 | 0.98 | 0.78 | 11 ms | 0 |
| hybrid + cross-encoder re-rank | **0.71** | 0.93 | **0.82** | 475 ms | 0 |
| multi-query | 0.67 | **1.00** | 0.81 | 1.7 s | 1 |
| HyDE | 0.67 | **1.00** | 0.81 | 2.0 s | 1 |
| *LangChain standard (1,000-char splitter)* | 0.50 | 0.91 | 0.67 | 43 ms | 0 |
| *LlamaIndex default (1,024-token chunks)* | 0.24 | 0.50 | 0.38 | 218 ms | 0 |
| *LlamaIndex, `chunk_size=256`* | 0.55 | 0.91 | 0.70 | 414 ms | 0 |

**Answers** (Claude Haiku 5.5 answers; Claude Sonnet 5.5 grades):

| Answers built on | Correct | Grounded | Cites the exact section | Refused off-topic | Wrongly refused |
|---|---|---|---|---|---|
| **vector** (default) | **0.95** | **0.81** | **1.00** | 6/6 | **0/42** |
| hybrid + re-rank | 0.95 | 0.57 | 0.91 | 6/6 | 1/42 |
| *LangChain standard* | 0.91 | 0.76 | 0.88 | 6/6 | 4/42 |
| *LlamaIndex default* | 0.52 | 0.81 | 0.50 | 6/6 | **16/42** |
| *LlamaIndex, `chunk_size=256`* | 0.88 | 0.76 | 0.88 | 6/6 | 1/42 |

All setups use the **same local embedding model**, so the framework rows measure each framework's
default chunking and retrieval, not the embeddings.

**What it shows**

0. **The hand-built pipeline beats both frameworks' standard setups, mainly through chunking**
   (split by headings into small chunks). And **a framework default can silently break retrieval:**
   LlamaIndex's 1,024-token chunks are 4× longer than the 256 tokens the embedding model reads, so 84% of
   chunks were searched by their first third only, and 16 of 42 real questions got "Not in docs". One
   setting (`chunk_size=256`) took correct answers from 0.52 to 0.88. **Match chunk size to the embedding model.**
1. **Among the hand-built methods, the differences are small.** Re-ranking gives the best top result; the methods
   that call Claude (multi-query, HyDE) add time and cost without beating plain vector search.
2. **Re-ranking hurt the answers:** equally correct, but much less grounded. The cross-encoder is
   trained on web search (MS MARCO), not manuals. **So `vector` is the default.**
3. **The remaining weakness is grounding, not retrieval:** about 1 answer in 5 adds helpful details
   that aren't in the sources (extra example code, table names). Next: a stricter prompt and a
   claim-level citation check.
4. **Refusing works:** every off-topic question was refused, and no real question was (vector).
5. **Evaluating the evaluator:** two bugs were found by reading outputs, not averages:
   - the judge was cut off before finishing its JSON, which counted as failures;
   - Claude's hidden "thinking" used up token limits, truncating HyDE passages, rephrasings and answers.

   That made HyDE look worst, multi-query look best, and short answers look more grounded. Both are
   fixed (thinking off for these short calls, a JSON-first judge), and every cut-off reply is now
   counted: 0 in the final run. The first run's answers are kept in `eval/results/answers_run1_truncated.json`.

**Caveat:** the in-scope questions were written by Claude *from* each section, so they're specific and
favour vector search. Real users type shorter, keyword-heavy questions where BM25 and hybrid usually
help. A hand-written question set is the next step.

## The six retrieval methods

| Method | How | Book chapter* |
|---|---|---|
| `vector` | embed the question (local `all-MiniLM-L6-v2`; Claude has no embedding model); cosine similarity with every chunk | ch07 |
| `bm25` | keyword scoring: rare words weigh more, long chunks slightly penalised | — |
| `hybrid` | both lists merged with **Reciprocal Rank Fusion**, keyed by section (fixes the book's position-keyed version, which never merged duplicates) | ch10 |
| `hybrid_rerank` | hybrid top 20 re-scored by a local cross-encoder reading question + chunk together | — |
| `multi_query` | Claude writes 3 rephrasings; hybrid for each; fused | ch09 |
| `hyde` | Claude writes a hypothetical answer; search with it, fused with the question's results | ch09 |

All search **chunks** but return their parent **sections** (ch08: small chunks to match, whole sections to answer).

**Framework comparisons** (`docsrag/frameworks.py`): `langchain` (`RecursiveCharacterTextSplitter` →
`InMemoryVectorStore` → retriever), `llamaindex` (`SentenceSplitter` defaults → `VectorStoreIndex` →
retriever) and `llamaindex_256`. Their chunks don't follow headings, so each retrieved chunk is mapped
back to the section(s) it covers, keeping the scoring identical.

\* *AI Agents and Applications: With LangChain, LangGraph, and MCP* (Roberto Infante, Manning).

## Experiment tracking with MLflow

Every `eval/run_eval.py` run is logged to MLflow: settings as **params**, one **nested run per method**
with its metrics, and the result files as **artifacts**. `--trace` also records every Claude call as an
MLflow **trace** (inputs, outputs, tokens, latency).

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

## MCP server

`mcp_server.py` uses each server feature from *AI Agents with MCP*:

| Feature | Here |
|---|---|
| Tools | `search_docs(query, method, k)`, `get_section(section_id)`, `refresh_book(book)` with **progress notifications** |
| Resources | `doc://books`, template `doc://{book}/{page}` |
| Prompt | `answer_with_citations(question)` |
| Completion | autocompletes book and page names for the resource template |
| Errors | `ToolError`, so the model reads what went wrong (e.g. an unknown section id, with similar ids) |

```bash
npx @modelcontextprotocol/inspector python mcp_server.py
claude mcp add docs-rag -- /path/to/python /full/path/to/mcp_server.py
```

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env               # LLM_API_KEY=sk-ant-...
python build_index.py              # download + index (~20 min the first time: the API rate-limits)
python build_index.py --no-fetch   # re-index the saved pages (~1 min)

python ask.py "How do I get the elements in either of two sets but not both?"
python ask.py --method bm25 --search-only "git stash pop"

python tests/test_offline.py       # offline: splitting, chunking, BM25, RRF
python eval/make_questions.py      # regenerate the test set (optional)
python eval/run_eval.py            # retrieval + answers (~20 min, ~200 Claude calls)
python eval/run_eval.py --methods llamaindex_256 --answer-methods llamaindex_256   # add one method (merged)
python eval/run_eval.py --trace    # also trace every Claude call in MLflow
python eval/run_eval.py --rejudge  # re-grade saved answers only
```

## Layout

```
docsrag/config.py      paths, models, Claude helper (thinking off, usage + truncation counters)
docsrag/ingest.py      Wikibooks API → sections → chunks (polite, resumable, progress callback)
docsrag/index.py       embeddings + BM25 (plain numpy)
docsrag/retrieve.py    the six methods, RRF, re-ranking
docsrag/frameworks.py  the same job with LlamaIndex and LangChain
docsrag/answer.py      cited answers or "Not in docs"
mcp_server.py          MCP server
ask.py                 command line
build_index.py         download + index
eval/                  question generation, evaluation (+ MLflow logging), results
notebooks/             walkthrough notebook
docs/HOW_IT_WORKS.md   ingestion, indexing and retrieval explained
tests/                 offline tests
```

## Data and licence

Text from [Wikibooks](https://en.wikibooks.org/): *Python Programming*, *Structured Query Language*
and *Git*, by their contributors, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The downloaded text and the index are **not** included in this repository; `build_index.py` fetches
them from the Wikibooks API. The page versions used for the results are listed in
`data/snapshot_revisions.json`. Short excerpts appear in the notebook outputs and evaluation results,
with the same attribution.

# How it works

From Wikibooks pages to a cited answer, in three parts:
**1. Ingestion** (`docsrag/ingest.py`), **2. Indexing** (`docsrag/index.py`) and **3. Retrieval** (`docsrag/retrieve.py`).

---

## 1. Ingestion: from a website to citable pieces

Ingestion does four jobs: **find pages → download them → split into sections → cut into chunks.**

### Job 1: find the pages (`list_pages`)

One request to the Wikibooks API lists every page whose title starts with the book name
(`Python_Programming/…`). A Wikibooks book is a main page plus sub-pages (`/Lists`, `/Loops`…).

`SKIP_TITLES` drops pages that **duplicate** content: *Print version*, *All Chapters*, *Contents*.
A "Print version" page holds the whole book; left in, it would match every question and make
citations and the evaluation meaningless.

### Job 2: download each page (`fetch_page`, `fetch_book`)

```python
{"action": "query", "prop": "extracts|info", "explaintext": 1, "exsectionformat": "wiki", "titles": title}
```

| Parameter | Why |
|---|---|
| `explaintext=1` | plain text: no HTML or wiki markup to clean |
| `exsectionformat=wiki` | keeps headings as `== Overview ==`, which job 3 needs |
| `prop=info` | the page's **revision id**; saved to `data/snapshot_revisions.json`, so results can be tied to the exact page versions |

**Polite to the API** (`_get`):

| Situation | What happens |
|---|---|
| normal | 0.6 s pause between pages |
| `429 Too Many Requests` | wait as long as the server's `Retry-After` header says |
| other errors | wait 3 s, 6 s, 12 s… (max 60 s), up to 7 tries |
| `maxlag=5` | asks Wikimedia's servers to say "wait" when they're busy |
| `User-Agent` | identifies the tool, as Wikimedia asks of bots |

The text is small (about 0.8 MB for all three books), but the API returns **one full page per
request** and rate-limits automated clients, so the first download takes about 20 minutes of
mostly waiting. It only happens once; `python build_index.py --no-fetch` re-indexes the saved pages.

**Resumable:** each page is appended to `data/raw/<book>.partial.jsonl` as it arrives. If a run is
interrupted, the next run skips pages it already has. When the book is complete, pages go to
`<book>.jsonl`. Pages under 200 characters (navigation stubs) are dropped.

**Progress:** `fetch_book(book, progress)` calls `progress(done, total, message)` after every page.
The terminal uses it for a progress line; the MCP server's `refresh_book` tool turns it into
**MCP progress notifications** (`ctx.report_progress`).

### Job 3: split pages into sections (`split_sections`)

Read the page line by line:

1. A line like `== Title ==` (regex `HEADING`, levels `==` to `======`) is a heading:
   save the text collected so far as a section, then update the **heading path** by level
   (`==` = 0, `===` = 1…), e.g. `List creation > List comprehensions`.
2. Any other line is added to the current section.
3. At the end, save the last section.

Text before the first heading becomes `Introduction`. A heading with no text of its own stays in
its children's path.

Each saved section gets:

- **an id**: `book/page#heading-path`, e.g. `python/Lists#list-creation-list-comprehensions`.
  This is what the chatbot **cites** and what the evaluation **scores**.
- **metadata**: book, page, heading path, revision id, text.

Dropped: sections under 80 characters, and *See also / References / External links / Notes / Sources*.

**Why headings, not fixed-size pieces?** The authors already split each topic into meaningful
parts. Cutting every N characters would split an example in half; headings keep one idea per section.

### Job 4: cut long sections into chunks (`chunk_section`)

- **Up to ~1,170 characters:** the section is **one chunk** (most sections).
- **Longer:** cut at **blank lines** into ~900-character chunks. Each new chunk repeats the last
  ~150 characters of the previous one (**overlap**), starting at a sentence or line boundary
  (`_overlap_tail`), so a sentence on a cut is still findable.
- **One huge paragraph** (e.g. a long code block): cut every ~900 characters.
- **Every chunk starts with its location**, a *contextual chunk header*:

```
Python Programming > Lists > List creation > List comprehensions
Using list comprehension, you describe the process using which the list should be created...
```

The chunk text alone might never say "Python" or "list"; the header lets search see the context.

### Example: *Python Programming / Lists* (19,858 characters)

```
26 sections, e.g.
   222 chars  Introduction
  1922 chars  Overview
  2081 chars  List creation > List comprehensions   → 3 chunks
  1718 chars  Getting pieces of lists (slices) > Continuous slices
```

### Result

| Level | What it's used for |
|---|---|
| **Pages** | the `doc://{book}/{page}` resource in the MCP server |
| **Sections** (parent) | what is cited and scored |
| **Chunks** (child) | what is searched: embeddings + BM25 |

**Search small, answer big:** small chunks match precisely; the answer gets the whole section.

---

## 2. Indexing: two ways to search the chunks (`docsrag/index.py`)

The 1,261 chunks get **two indexes**: a **vector index** (meaning) and a **BM25 index** (keywords).

### Step 1: collect sections and chunks (`build`)

```python
for key in BOOKS:                       # python, sql, git
    for page in load_raw(key):          # downloaded pages
        for sec in split_sections(key, page):
            sections.append(sec)
            for text in chunk_section(sec):
                chunks.append({"chunk_id": len(chunks), "section_id": sec["id"], "text": text})
```

Each chunk gets a **number** and remembers its **section**: the link that makes parent/child
retrieval work. Result: **926 sections, 1,261 chunks**.

### Step 2: turn each chunk into a vector (embedding)

```python
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True)
```

- **An embedding** is a list of **384 numbers** describing what the text *means*. Texts about
  similar things get similar numbers, even with different words ("undo a commit" ≈ "revert a change").
- **The model** `all-MiniLM-L6-v2` is small (~90 MB), free and **runs locally**; needed because
  Claude has no embedding model.
- Embedded **256 chunks at a time**; all 1,261 take under a minute on a laptop CPU.
- **Normalised** to length 1, so comparing two vectors is a dot product = **cosine similarity**.
- Saved as `data/index/embeddings.npy`: a 1,261 × 384 table of numbers (~1.9 MB).

### Step 3: save the text side

- `data/index/sections.jsonl`: the 926 sections (id, book, page, heading, text, revision).
- `data/index/chunks.jsonl`: the 1,261 chunks (number, section id, text with header).

**Row *n* of `embeddings.npy` is the vector of chunk *n***. That's how a vector match leads back
to its chunk, and then to its section.

### Step 4: the keyword index (BM25), built when the index loads

Not saved; rebuilt in about a second on load:

1. **Tokenize** each chunk: lower-case, keep words and code-like tokens (`append`, `git_reset`),
   drop common words ("the", "how", "use"…).
2. **Count words per chunk** (`tf`).
3. **Measure how rare each word is** across all chunks (`idf`): `stash` is rare (high weight),
   `list` is common (low weight).
4. **Record chunk lengths**, so long chunks don't win just by containing more words.

A chunk's score = sum over the query's words of *rarity × dampened count ÷ length adjustment*:
the standard Okapi BM25 formula (`k1 = 1.5`, `b = 0.75`), written in ~20 readable lines.

### Step 5: load once, reuse (`Index.load`)

```python
@lru_cache(maxsize=1)
def load(): ...   # reads the three files, builds BM25 and the section -> chunks map
```

- Loaded **once per run** and kept in memory: BM25 search ~2 ms, vector search ~150 ms
  (mostly embedding the question).
- **`chunks_of`**: section id → its chunk numbers; the re-ranker uses it to score each candidate
  section's chunks.

### How a search uses the index

| Search | What runs |
|---|---|
| **vector** | embed the question (384 numbers) → `embeddings @ question` (1,261 dot products in one matrix multiply) → top 60 chunks → their sections |
| **bm25** | tokenize the question → score every chunk → top 60 chunks → their sections |

### Why no vector database (Chroma, Pinecone…)?

- **The data is small:** 1,261 × 384 numbers fit in memory and comparing against all of them takes
  milliseconds. A vector database pays off at hundreds of thousands of chunks, where it uses
  *approximate* search to avoid comparing everything.
- **Transparent:** every step is a few lines of numpy. The LangChain book used Chroma
  (chapters 6–10); this shows what Chroma does underneath.
- **Swappable:** at a larger scale only `index.py` changes (Chroma, pgvector, FAISS). Retrieval,
  answering, the MCP server and the evaluation stay the same.

### When it rebuilds

| Command | Does |
|---|---|
| `python build_index.py` | download, then index |
| `python build_index.py --no-fetch` | re-index the saved pages (e.g. after changing the chunking) |
| MCP tool `refresh_book` | re-download one book and rebuild, with progress notifications |

---

## 3. Retrieval: six methods

Every method takes a question and returns ranked **section** ids (found via their chunks).

| Method | How | Strong at | Claude calls |
|---|---|---|---|
| `vector` | embed the question, cosine similarity with every chunk | different wording ("undo" ≈ "revert") | 0 |
| `bm25` | keyword scoring: rare words count more, long chunks slightly penalised | exact names (`git stash pop`, `LEFT JOIN`) | 0 |
| `hybrid` | vector + BM25 merged with **Reciprocal Rank Fusion** | both kinds of question | 0 |
| `hybrid_rerank` | hybrid top 20, re-scored by a local **cross-encoder** that reads question and chunk together | precision at the top | 0 |
| `multi_query` | Claude writes 3 rephrasings; hybrid for each; fused with RRF | vague questions | 1 |
| `hyde` | Claude writes a hypothetical answer; vector search with it, fused with the question's vector + BM25 results | short or vague questions | 1 |

**Child → parent:** a section's rank is the rank of its best chunk (`to_sections`).

**Reciprocal Rank Fusion:** each section scores `1 / (60 + rank)` in each list and the scores are
added, so a section ranked well by several lists rises to the top. Results are fused **by section
id**. (The LangChain book's chapter 10 version kept results by position, so the same document from
two lists was never merged; `tests/test_offline.py::test_rrf_merges_same_item` checks the fix.)

**Re-ranking:** vector search turns the question and the text into numbers *separately*; a
cross-encoder reads them *side by side*, which is more precise but slower, so it only re-scores
a shortlist. Retrieve wide, then re-rank narrow.

### Answering (`docsrag/answer.py`)

1. Take the top 5 sections from the chosen method.
2. Give them to Claude numbered `[1]`–`[5]`, with rules: use only these sections, cite every fact
   like `[2]`, and if the answer isn't there, start with **"Not in docs:"**.
3. Record which sections were cited (for the citation-accuracy score).

---

## 4. The same job with frameworks

`docsrag/frameworks.py` builds the same retrieval with each framework's standard pipeline, using the
**same embedding model**:

| | LlamaIndex | LangChain |
|---|---|---|
| Unit of text | `Document` (a page) → `Node` (a chunk) | `Document` → split `Document`s |
| Splitter | `SentenceSplitter` (default 1,024 tokens, 200 overlap) | `RecursiveCharacterTextSplitter(1000, 200)` |
| Embeddings | `Settings.embed_model = HuggingFaceEmbedding(...)` | `HuggingFaceEmbeddings(...)` |
| Store | `VectorStoreIndex` (persisted with `storage_context.persist`) | `InMemoryVectorStore` (`dump` / `load`) |
| Search | `index.as_retriever(similarity_top_k=n).retrieve(q)` | `store.as_retriever(search_kwargs={"k": n}).invoke(q)` |

Their chunks ignore headings, so each retrieved chunk is mapped back to the section(s) it covers
(`chunk_to_sections`, by word 5-gram overlap) and scored exactly like the hand-built methods.

**Chunk size must match the embedding model.** `all-MiniLM-L6-v2` reads only the first 256 tokens of a
text. LlamaIndex's default 1,024-token chunks (sized for OpenAI embeddings) left 84% of chunks mostly
invisible; `SentenceSplitter(chunk_size=256)` fixes it.

## 5. Tracking experiments (MLflow)

`eval/run_eval.py` logs each evaluation to MLflow (`sqlite:///mlflow.db`, experiment `docs-rag-eval`):
params (models, number of questions), one nested run per method with its metrics, and the result files
as artifacts. Runs that evaluate only some methods are **merged** into the saved results, so the latest
MLflow run always holds every method. `--trace` turns on `mlflow.anthropic.autolog()`, recording each
Claude call as a trace span.

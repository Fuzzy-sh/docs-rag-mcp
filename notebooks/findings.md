## 7. Findings

**1. Every method finds the right *page*; on these questions the methods are close.**
page@5 is 1.00 for every method. Re-ranking gives the best top result (hit@1 0.71, MRR 0.82); `vector`,
`multi_query` and `hyde` never miss the top 5. The two methods that call Claude (`multi_query`, `hyde`)
add ~1.7–2 s and a paid call per question without beating plain vector search here.

**2. Hybrid did not beat plain vector search, and re-ranking hurt the answers.**
BM25 misses questions phrased in everyday words, and fusion pulls those sections down (the
*"undo my last commit"* example in section 3). The cross-encoder raised hit@1 but dropped some correct
sections out of the top 5, and answers built on its results were **less grounded (0.57 vs 0.81)**, with
equal correctness (0.95). The re-ranker is trained on web search (MS MARCO), not manuals.
**So the default method is `vector`:** the best answers, no extra Claude calls.

**3. The main remaining weakness is grounding, not retrieval.** Answers are correct 95% of the time and
cite the exact gold section (vector: 100%), but about 1 in 5 adds "helpful" details that aren't in the
sources: extra example code, a table name, *"Python packs these into a tuple"*. The judge marks those
claims unsupported. **Next:** a stricter prompt (no code or facts beyond the sections), and a check that
every cited claim appears in its section.

**4. Refusing works.** All 6 off-topic questions were refused, and almost no real question was
(0/42 for vector, 1/42 for re-rank, where the right section wasn't retrieved).

**5. Evaluating the evaluator: two bugs found by looking at the outputs, not the averages.**
- **The judge** reasoned before its JSON and was cut off at 200 tokens; cut-off verdicts counted as
  failures (first scores: vector 0.48, re-rank 0.33 correct). Fixed with a JSON-first prompt and room to
  finish; saved answers were re-graded, not regenerated.
- **Claude's hidden "thinking" used up the token limits**, so HyDE passages, multi-query rephrasings and
  answers were often cut short or empty. That made HyDE look worst (MRR 0.73 → **0.81** once fixed) and
  multi-query look best (hit@1 0.76 → **0.67**), and made truncated answers look *more* grounded and
  produced false "Not in docs" replies. Fixed by turning thinking off for these short calls (models
  differ: `disabled` vs `between_tools`) and **counting every reply that hits the limit** (0 in the final run).

**Caveat:** the in-scope questions were written by Claude *from* each section, so they're specific and
favour vector search (it already reaches hit@5 = 1.00). Real users type shorter, keyword-heavy
questions (error messages, command names) where BM25 and hybrid usually help. **Next:** a
hand-written question set.

| | Best on this test set |
|---|---|
| Top-ranked section | `hybrid_rerank` (hit@1 0.71, MRR 0.82) |
| Never missing the top 5 | `vector`, `multi_query`, `hyde` (hit@5 1.00) |
| Answers (correct / grounded) | `vector` (0.95 / 0.81) |
| Speed | `bm25` (2 ms) |
| Refusing off-topic questions | all methods (6/6) |

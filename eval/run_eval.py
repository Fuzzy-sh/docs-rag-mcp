"""Evaluate the six retrieval methods, then the chatbot's answers.

  python eval/run_eval.py                     # retrieval for all methods (+ answers, see below)
  python eval/run_eval.py --skip-answers      # retrieval only
  python eval/run_eval.py --answer-methods vector,hybrid_rerank

RETRIEVAL (in-scope questions; gold = the section the question was written from)
  hit@1, hit@5   the gold section is ranked 1st / in the top 5
  page@5         a section from the gold PAGE is in the top 5 (the right chapter, maybe a sibling heading)
  MRR@10         1 / rank of the gold section (0 if not in the top 10), averaged
  ms, LLM calls  average latency and Claude calls per question

ANSWERS (all questions, top-5 sections from the given method)
  correct        a judge model says the answer matches the reference answer
  grounded       the judge says every claim is supported by the retrieved sections
  cites gold     the answer cites the gold section (or a section of the gold page)
  refusals       out-of-scope questions answered "Not in docs" (higher is better)
  false refusals in-scope questions wrongly answered "Not in docs" (lower is better)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docsrag.answer import answer, format_sources  # noqa: E402
from docsrag.config import JUDGE_MODEL, LLM_MODEL, USAGE, llm  # noqa: E402
from docsrag.retrieve import METHODS, search  # noqa: E402

JUDGE = """You are grading a documentation chatbot.

Retrieved sections:
{sources}

Question: {question}
Reference answer: {reference}
Chatbot answer: {answer}

Grade two things:
- correct: does the chatbot answer give the same key information as the reference answer?
  (Extra correct detail is fine. A "Not in docs" reply is not correct.)
- grounded: is every factual claim in the chatbot answer supported by the retrieved sections?
Reply with ONLY this JSON object, nothing before or after it, and keep "why" under 25 words:
{{"correct": true or false, "grounded": true or false, "why": "short reason"}}"""


def parse_verdict(text: str) -> dict:
    """Read the judge's JSON; if it's malformed (e.g. unescaped quotes in 'why'), read the two flags directly."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    flag = lambda k: bool(re.search(rf'"{k}"\s*:\s*true', text, re.I))
    return {"correct": flag("correct"), "grounded": flag("grounded"), "why": text[:300], "parsed": "fallback"}


def retrieval_eval(questions: list[dict], methods: list[str]) -> dict:
    rows, summary = [], {}
    inscope = [q for q in questions if q["in_scope"]]
    for m in methods:
        hit1 = hit5 = page5 = mrr = ms = calls = 0
        for q in inscope:
            before = USAGE["calls"]
            t0 = time.time()
            ranked = search(q["question"], m, k=10)
            ms += (time.time() - t0) * 1000
            calls += USAGE["calls"] - before
            rank = ranked.index(q["gold_section"]) + 1 if q["gold_section"] in ranked else None
            on_page = any(s.split("#")[0] == q["gold_page"] for s in ranked[:5])
            hit1 += rank == 1
            hit5 += bool(rank and rank <= 5)
            page5 += on_page
            mrr += 1 / rank if rank else 0
            rows.append({"method": m, "id": q["id"], "rank": rank, "page@5": on_page, "top5": ranked[:5]})
        n = len(inscope)
        summary[m] = {"hit@1": round(hit1 / n, 3), "hit@5": round(hit5 / n, 3), "page@5": round(page5 / n, 3),
                      "MRR@10": round(mrr / n, 3), "avg_ms": round(ms / n), "llm_calls_per_q": round(calls / n, 1)}
        s = summary[m]
        print(f"  {m:14} hit@1 {s['hit@1']:.2f}  hit@5 {s['hit@5']:.2f}  page@5 {s['page@5']:.2f}  "
              f"MRR {s['MRR@10']:.2f}  {s['avg_ms']:>5} ms  {s['llm_calls_per_q']} LLM calls")
    return {"summary": summary, "rows": rows}


def answer_eval(questions: list[dict], methods: list[str]) -> dict:
    rows, summary = [], {}
    for m in methods:
        st = dict(correct=0, grounded=0, cites=0, cites_page=0, refusals=0, false_refusals=0, n_in=0, n_out=0)
        for q in questions:
            r = answer(q["question"], m)
            row = {"method": m, "id": q["id"], "in_scope": q["in_scope"], "answer": r["answer"],
                   "not_in_docs": r["not_in_docs"], "cited": r["cited"], "sources": r["sources"]}
            if q["in_scope"]:
                st["n_in"] += 1
                st["false_refusals"] += r["not_in_docs"]
                cites = q["gold_section"] in r["cited"]
                st["cites"] += cites
                st["cites_page"] += any(c.split("#")[0] == q["gold_page"] for c in r["cited"])
                verdict = llm(JUDGE.format(sources=format_sources(r["sources"]), question=q["question"],
                                           reference=q["reference_answer"], answer=r["answer"]),
                              model=JUDGE_MODEL, max_tokens=800)
                v = parse_verdict(verdict)
                st["correct"] += bool(v.get("correct"))
                st["grounded"] += bool(v.get("grounded"))
                row.update(cites_gold=cites, judge=v)
            else:
                st["n_out"] += 1
                st["refusals"] += r["not_in_docs"]
            rows.append(row)
        n_in, n_out = st["n_in"], st["n_out"]
        summary[m] = {"correct": round(st["correct"] / n_in, 3), "grounded": round(st["grounded"] / n_in, 3),
                      "cites_gold": round(st["cites"] / n_in, 3), "cites_gold_page": round(st["cites_page"] / n_in, 3),
                      "refusals_out_of_scope": f"{st['refusals']}/{n_out}",
                      "false_refusals": f"{st['false_refusals']}/{n_in}"}
        s = summary[m]
        print(f"  {m:14} correct {s['correct']:.2f}  grounded {s['grounded']:.2f}  cites gold {s['cites_gold']:.2f}  "
              f"refused out-of-scope {s['refusals_out_of_scope']}  false refusals {s['false_refusals']}")
    return {"summary": summary, "rows": rows}


def rejudge(questions: list[dict]) -> dict:
    """Re-grade the saved answers in eval/results/answers.json (no new answers are generated)."""
    path = ROOT / "eval" / "results" / "answers.json"
    saved = json.loads(path.read_text())
    qs = {q["id"]: q for q in questions}
    summary = {}
    for m in dict.fromkeys(r["method"] for r in saved["rows"]):
        rows = [r for r in saved["rows"] if r["method"] == m]
        st = dict(correct=0, grounded=0, cites=0, cites_page=0, refusals=0, false_refusals=0, n_in=0, n_out=0)
        for r in rows:
            q = qs[r["id"]]
            if not q["in_scope"]:
                st["n_out"] += 1
                st["refusals"] += r["not_in_docs"]
                continue
            st["n_in"] += 1
            st["false_refusals"] += r["not_in_docs"]
            v = parse_verdict(llm(JUDGE.format(sources=format_sources(r["sources"]), question=q["question"],
                                               reference=q["reference_answer"], answer=r["answer"]),
                                  model=JUDGE_MODEL, max_tokens=800))
            r["judge"], r["cites_gold"] = v, q["gold_section"] in r["cited"]
            st["correct"] += bool(v.get("correct"))
            st["grounded"] += bool(v.get("grounded"))
            st["cites"] += r["cites_gold"]
            st["cites_page"] += any(c.split("#")[0] == q["gold_page"] for c in r["cited"])
        n_in, n_out = st["n_in"], st["n_out"]
        summary[m] = {"correct": round(st["correct"] / n_in, 3), "grounded": round(st["grounded"] / n_in, 3),
                      "cites_gold": round(st["cites"] / n_in, 3), "cites_gold_page": round(st["cites_page"] / n_in, 3),
                      "refusals_out_of_scope": f"{st['refusals']}/{n_out}",
                      "false_refusals": f"{st['false_refusals']}/{n_in}"}
        s = summary[m]
        print(f"  {m:14} correct {s['correct']:.2f}  grounded {s['grounded']:.2f}  cites gold {s['cites_gold']:.2f}  "
              f"refused out-of-scope {s['refusals_out_of_scope']}  false refusals {s['false_refusals']}")
    saved["summary"] = summary
    path.write_text(json.dumps(saved, indent=1))
    return saved


def merge(path: Path, new: dict) -> dict:
    """Merge results for the methods just run into the saved file (other methods are kept)."""
    if not path.exists():
        return new
    old = json.loads(path.read_text())
    ran = set(new["summary"])
    old["summary"] = {**{m: v for m, v in old["summary"].items() if m not in ran}, **new["summary"]}
    old["rows"] = [r for r in old["rows"] if r["method"] not in ran] + new["rows"]
    return old


def _frac(v) -> float:
    """'6/6' -> 1.0; numbers pass through."""
    if isinstance(v, str) and "/" in v:
        a, b = v.split("/")
        return int(a) / int(b)
    return float(v)


def log_to_mlflow(out_dir: Path, questions: list[dict], ran: list[str]) -> None:
    """One MLflow run per evaluation: settings as params, one nested run per method with its
    metrics, and the result files as artifacts. View with:  mlflow ui --backend-store-uri sqlite:///mlflow.db"""
    import mlflow
    from docsrag.config import EMBED_MODEL, RERANK_MODEL

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("docs-rag-eval")
    ret = json.loads((out_dir / "retrieval.json").read_text()) if (out_dir / "retrieval.json").exists() else {"summary": {}}
    ans = json.loads((out_dir / "answers.json").read_text()) if (out_dir / "answers.json").exists() else {"summary": {}}
    with mlflow.start_run(run_name=f"eval ({', '.join(ran)})"):
        mlflow.log_params({"answer_model": LLM_MODEL, "judge_model": JUDGE_MODEL, "embed_model": EMBED_MODEL,
                           "rerank_model": RERANK_MODEL, "questions": len(questions),
                           "in_scope": sum(q["in_scope"] for q in questions), "methods_run_now": ",".join(ran)})
        mlflow.log_metrics({"claude_calls": USAGE["calls"], "claude_truncated": USAGE["truncated"]})
        for m in sorted(set(ret["summary"]) | set(ans["summary"])):
            with mlflow.start_run(run_name=m, nested=True):
                mlflow.log_param("method", m)
                for k, v in ret["summary"].get(m, {}).items():
                    mlflow.log_metric(f"retrieval_{k.replace('@', '_at_')}", _frac(v))
                for k, v in ans["summary"].get(m, {}).items():
                    mlflow.log_metric(f"answer_{k}", _frac(v))
        for f in ("retrieval.json", "answers.json"):
            if (out_dir / f).exists():
                mlflow.log_artifact(str(out_dir / f))
        mlflow.log_artifact(str(ROOT / "eval" / "questions.jsonl"))
    print("Logged to MLflow (experiment 'docs-rag-eval'). View: mlflow ui --backend-store-uri sqlite:///mlflow.db")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--skip-answers", action="store_true")
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--rejudge", action="store_true", help="re-grade saved answers only")
    ap.add_argument("--answer-methods", default="vector,hybrid_rerank,llamaindex,langchain")
    ap.add_argument("--no-mlflow", action="store_true", help="don't log this run to MLflow")
    ap.add_argument("--trace", action="store_true", help="also record every Claude call as an MLflow trace")
    a = ap.parse_args()

    if a.trace:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        mlflow.set_experiment("docs-rag-traces")
        mlflow.anthropic.autolog()          # every Anthropic SDK call becomes a trace span

    questions = [json.loads(l) for l in open(ROOT / "eval" / "questions.jsonl")]
    if a.rejudge:
        print("RE-JUDGING saved answers")
        rejudge(questions)
        print(f"\nClaude usage: {USAGE['calls']} calls")
        return
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(exist_ok=True)
    print(f"{len(questions)} questions ({sum(q['in_scope'] for q in questions)} in scope). "
          f"Answer model: {LLM_MODEL}, judge: {JUDGE_MODEL}\n\nRETRIEVAL")
    ran = []
    if not a.skip_retrieval:
        ret = retrieval_eval(questions, a.methods.split(","))
        (out_dir / "retrieval.json").write_text(json.dumps(merge(out_dir / "retrieval.json", ret), indent=1))
        ran += a.methods.split(",")
    if not a.skip_answers:
        print("\nANSWERS")
        ans = answer_eval(questions, a.answer_methods.split(","))
        (out_dir / "answers.json").write_text(json.dumps(merge(out_dir / "answers.json", ans), indent=1))
        ran += [m for m in a.answer_methods.split(",") if m not in ran]
    print(f"\nClaude usage: {USAGE['calls']} calls, {USAGE['input_tokens']:,} in / {USAGE['output_tokens']:,} out tokens, "
          f"{USAGE['truncated']} replies hit max_tokens")
    if not a.no_mlflow:
        log_to_mlflow(out_dir, questions, ran)


if __name__ == "__main__":
    main()

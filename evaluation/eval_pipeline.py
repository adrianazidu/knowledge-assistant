"""
evaluation/eval_pipeline.py — test any pipeline configuration.

Evaluates the system with three scoring methods:
  keyword  — free, instant
  semantic — embedding similarity
  judge    — GPT-4o as evaluator (most accurate)

Usage:
  from evaluation.eval_pipeline import run
  report = run()

  Or standalone:
  python -m evaluation.eval_pipeline
  python -m evaluation.eval_pipeline --no-judge --category shipping
"""
import json, os, sys, argparse
from datetime import datetime
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI
from inference.pipeline import ask as rag_ask
import config


# ── Runner ────────────────────────────────────────────────────────────────────

def _get_answer(question: str, client: OpenAI) -> str:
    if config.EVAL_TARGET == "rag":
        return rag_ask(question)["answer"]
    model = config.OPENAI_CHAT_MODEL if config.EVAL_TARGET == "openai" else config.VLLM_CHAT_MODEL
    base  = None if config.EVAL_TARGET == "openai" else config.VLLM_CHAT_URL
    c = OpenAI(api_key=config.OPENAI_API_KEY, **({"base_url": base} if base else {}))
    resp = c.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":config.SUPPORT_SYSTEM_PROMPT},
                  {"role":"user","content":question}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ── Evaluators ────────────────────────────────────────────────────────────────

def _keyword_score(actual: str, keywords: list[str]) -> dict:
    if not keywords:
        return {"score": 1.0, "passed": True, "details": "no keywords"}
    al = actual.lower()
    found   = [k for k in keywords if k.lower() in al]
    missing = [k for k in keywords if k.lower() not in al]
    score   = len(found) / len(keywords)
    return {"score": round(score,3), "passed": score >= config.PASS_THRESHOLD,
            "found": found, "missing": missing}

def _embed(text: str, client: OpenAI) -> np.ndarray:
    r = client.embeddings.create(input=text, model="text-embedding-3-small")
    return np.array(r.data[0].embedding)

def _semantic_score(actual: str, expected: str, client: OpenAI) -> dict:
    a, b  = _embed(actual, client), _embed(expected, client)
    score = float(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"score": round(score,3), "passed": score >= config.PASS_THRESHOLD}

JUDGE_TMPL = """Evaluate this AI answer.
Question: {q}
Expected: {e}
Actual:   {a}

Reply ONLY with JSON:
{{"correctness":0-1,"completeness":0-1,"hallucination":0-1,"overall_score":0-1,"passed":true/false,"reason":"one sentence"}}"""

def _judge_score(question: str, actual: str, expected: str, client: OpenAI) -> dict:
    resp = client.chat.completions.create(
        model=config.JUDGE_MODEL,
        messages=[{"role":"user","content":JUDGE_TMPL.format(q=question,e=expected,a=actual)}],
        temperature=0, response_format={"type":"json_object"},
    )
    r = json.loads(resp.choices[0].message.content)
    return {k: round(float(r[k]),3) if isinstance(r[k], (int,float)) else r[k]
            for k in ("correctness","completeness","hallucination","overall_score","passed","reason")}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(use_semantic=True, use_judge=True, category=None, run_name=None) -> dict:
    print(f"\n╔══════════════════════════════════════╗")
    print(f"║         EVALUATION PIPELINE           ║")
    print(f"╚══════════════════════════════════════╝")
    print(f"  Target: {config.EVAL_TARGET} | Judge: {config.JUDGE_MODEL if use_judge else 'off'}")

    with open(config.TEST_CASES_PATH) as f:
        cases = json.load(f)
    if category:
        cases = [c for c in cases if c.get("category") == category]

    client  = OpenAI(api_key=config.OPENAI_API_KEY)
    results = []

    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']}...", end="\r")
        actual  = _get_answer(case["question"], client)
        scores  = {"keyword": _keyword_score(actual, case.get("expected_keywords",[]))}
        if use_semantic:
            scores["semantic"] = _semantic_score(actual, case["expected_answer"], client)
        if use_judge:
            scores["llm_judge"] = _judge_score(case["question"], actual,
                                               case["expected_answer"], client)

        overall = (scores.get("llm_judge") or scores.get("semantic") or scores["keyword"])
        results.append({**case, "actual_answer": actual, "scores": scores,
                        "overall_score": overall["score"], "passed": overall["passed"]})

    print(f"  Scored {len(results)} cases.          ")

    # Aggregate
    passed    = sum(1 for r in results if r["passed"])
    pass_rate = passed / len(results)
    by_cat    = defaultdict(lambda: {"total":0,"passed":0})
    for r in results:
        c = r.get("category","?")
        by_cat[c]["total"]  += 1
        by_cat[c]["passed"] += 1 if r["passed"] else 0

    report = {
        "run_name":  run_name or f"{config.EVAL_TARGET}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "target":    config.EVAL_TARGET,
        "summary": {
            "total": len(results), "passed": passed,
            "pass_rate": round(pass_rate,3),
            "ci_gate": "PASS" if pass_rate >= config.PASS_THRESHOLD else "FAIL",
        },
        "by_category": {
            c: {"total":v["total"],"passed":v["passed"],
                "pass_rate":round(v["passed"]/v["total"],3)}
            for c,v in by_cat.items()
        },
        "results": results,
    }

    # Print
    g = "✅" if report["summary"]["ci_gate"] == "PASS" else "❌"
    s = report["summary"]
    print(f"\n  {g} CI Gate: {s['ci_gate']} | Pass rate: {s['pass_rate']:.1%} ({s['passed']}/{s['total']})")
    for cat, v in report["by_category"].items():
        icon = "✅" if v["pass_rate"] >= config.PASS_THRESHOLD else "❌"
        print(f"  {icon} {cat:<18} {v['pass_rate']:.0%} ({v['passed']}/{v['total']})")
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n  Failed ({len(failures)}):")
        for r in failures:
            reason = r["scores"].get("llm_judge",{}).get("reason","")
            print(f"  ❌ {r['id']}: {r['question'][:50]}")
            if reason: print(f"     → {reason}")

    # Save
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"{report['run_name']}.json")
    with open(path,"w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report → {path}\n")

    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-judge",    action="store_true")
    p.add_argument("--no-semantic", action="store_true")
    p.add_argument("--category",    default=None)
    a = p.parse_args()
    import sys
    r = run(use_semantic=not a.no_semantic, use_judge=not a.no_judge, category=a.category)
    sys.exit(0 if r["summary"]["ci_gate"]=="PASS" else 1)

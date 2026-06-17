"""
main.py — unified entry point for the knowledge assistant AI project.

Every pipeline is accessible from here with a single command.

QUICK START:
  1. Add keys to .env
  2. Generate synthetic data with finetune_data/generate.py and add docs to data/raw/ and/or set GitLab vars in .env
  3. python main.py ingest --source local
  4. python main.py chat

COMMANDS:

  ingest   — build / refresh the vector store
    python main.py ingest --source local
    python main.py ingest --source gitlab
    python main.py ingest --source local gitlab --reset
    python main.py ingest --source gitlab --no-issues --branch main
    python main.py ingest --stats

  chat     — RAG question answering (no agent, just retrieval + LLM)
    python main.py chat
    python main.py chat --question "How does the payment service work?"
    python main.py chat --filter code          # only search code chunks
    python main.py chat --debug                # show retrieved sources

  agent    — full agentic mode (tools + RAG + order lookup + tickets)
    python main.py agent
    python main.py agent --demo                # run preset scenarios
    python main.py agent --scenario 2

  finetune — fine-tuning pipeline
    python main.py finetune prep               # format training data
    python main.py finetune prep --preview     # preview one formatted example
    python main.py finetune train              # run LoRA training (needs GPU)
    python main.py finetune train --dry-run    # verify setup without training
    python main.py finetune serve              # serve fine-tuned model via vLLM
    python main.py finetune serve --mode lora  # serve with hot-swappable adapters

  eval     — evaluation pipeline
    python main.py eval                        # full eval (keyword + semantic + judge)
    python main.py eval --no-judge             # fast mode
    python main.py eval --category returns     # one category
    python main.py eval --target rag           # eval the RAG pipeline
    python main.py eval --target vllm          # eval your fine-tuned model
    python main.py eval --list                 # show saved results
    python main.py eval --compare a.json b.json

  status   — show what's built and ready
    python main.py status
"""

import argparse
import sys
import os
import json


# ── status ────────────────────────────────────────────────────────────────────

def cmd_status():
    import config
    print(f"\n  {config.PROJECT_NAME} — System Status")
    print("  " + "─"*45)

    # Vector store
    try:
        from ingestion import vector_store
        s = vector_store.stats()
        if s["total"] > 0:
            print(f"  ✅ Vector store:   {s['total']} chunks")
            for t, n in s.get("by_type",{}).items():
                print(f"     {t:<18} {n}")
        else:
            print(f"  ❌ Vector store:   empty — run: python main.py ingest")
    except Exception as e:
        print(f"  ❌ Vector store:   error ({e})")

    # Fine-tuned model
    lora   = os.path.exists(config.FINAL_MODEL_DIR)
    merged = os.path.exists(config.MERGED_MODEL_DIR)
    if merged:
        print(f"  ✅ Fine-tuned model (merged): {config.MERGED_MODEL_DIR}")
    elif lora:
        print(f"  ✅ Fine-tuned model (LoRA):   {config.FINAL_MODEL_DIR}")
    else:
        print(f"  ⚠️  Fine-tuned model: not trained (using {config.OPENAI_CHAT_MODEL})")

    # Training data
    tp = config.FORMATTED_DATA_PATH.replace(".jsonl","_train.jsonl")
    if os.path.exists(tp):
        with open(tp) as f:
            n = sum(1 for _ in f)
        print(f"  ✅ Training data:   {n} formatted examples")
    else:
        print(f"  ⚠️  Training data:  not prepared — run: python main.py finetune prep")

    # Test cases
    if os.path.exists(config.TEST_CASES_PATH):
        with open(config.TEST_CASES_PATH) as f:
            n = len(json.load(f))
        print(f"  ✅ Test cases:      {n} cases in {config.TEST_CASES_PATH}")
    else:
        print(f"  ❌ Test cases:      not found at {config.TEST_CASES_PATH}")

    # Eval results
    if os.path.exists(config.RESULTS_DIR):
        runs = [f for f in os.listdir(config.RESULTS_DIR) if f.endswith(".json")]
        if runs:
            latest = sorted(runs)[-1]
            with open(os.path.join(config.RESULTS_DIR, latest)) as f:
                r = json.load(f)
            s = r["summary"]
            g = "✅" if s["ci_gate"]=="PASS" else "❌"
            print(f"  {g} Last eval:       {s['pass_rate']:.0%} pass rate ({latest[:20]}...)")

    # Backends
    print(f"\n  Config:")
    print(f"  Embedding:  {config.EMBEDDING_BACKEND}")
    print(f"  Inference:  {config.MODEL_BACKEND} ({config.OPENAI_CHAT_MODEL if config.MODEL_BACKEND=='openai' else config.VLLM_CHAT_MODEL})")
    print(f"  Eval target:{config.EVAL_TARGET}")
    print()


# ── ingest ────────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    from ingestion.ingest import run
    from ingestion import vector_store
    if args.stats:
        s = vector_store.stats()
        print(f"\n  Chunks: {s['total']} | Types: {s.get('by_type',{})}\n")
    else:
        run(sources=args.source, reset=args.reset, gitlab_branch=args.branch,
            gitlab_extensions=args.extensions, include_issues=not args.no_issues,
            include_mrs=not args.no_mrs, include_wiki=not args.no_wiki)


# ── chat ──────────────────────────────────────────────────────────────────────

def cmd_chat(args):
    from inference.pipeline import ask

    if args.question:
        result = ask(args.question, type_filter=args.filter)
        print(f"\n  Answer: {result['answer']}")
        if args.debug:
            print(f"\n  Sources:")
            for s in result["sources"]:
                print(f"    [{s['score']:.2f}] {s['source']}")
                print(f"    {s['text']}")
        return

    print("\n  RAG Chat — type 'quit' to exit | 'debug on/off' to toggle sources")
    debug = args.debug
    while True:
        try:
            q = input("\n  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q or q.lower() == "quit":
            break
        if q.lower() == "debug on":
            debug = True; continue
        if q.lower() == "debug off":
            debug = False; continue

        result = ask(q, type_filter=args.filter)
        print(f"\n  Answer: {result['answer']}")
        if debug:
            print(f"\n  Sources:")
            for s in result["sources"]:
                print(f"    [{s['score']:.2f}] {s['source']}")


# ── agent ─────────────────────────────────────────────────────────────────────

DEMO_SCENARIOS = [
    {"name":"Order lookup",          "messages":["Can you check order ORD-1002?"]},
    {"name":"Return + refund",       "messages":["I want to return order ORD-1004 and get a refund"]},
    {"name":"Policy question",       "messages":["What payment methods do you accept?"]},
    {"name":"Technical question",    "messages":["How does the payment retry logic work in the codebase?"]},
    {"name":"Multi-turn",            "messages":["I need help","It's about order ORD-1001","Can I return it?"]},
    {"name":"Escalation",            "messages":["I want to return ORD-1001. I bought it months ago. Email is maria@example.com"]},
]

def cmd_agent(args):
    from agents.agent import run as agent_run

    if args.demo or args.scenario:
        scenarios = ([DEMO_SCENARIOS[args.scenario-1]] if args.scenario
                     else DEMO_SCENARIOS)
        for i, s in enumerate(scenarios, 1):
            print(f"\n{'═'*55}")
            print(f"  Scenario {i}: {s['name']}")
            print(f"{'═'*55}")
            history = []
            for msg in s["messages"]:
                result  = agent_run(msg, history, verbose=True)
                history = result["history"]
            print(f"\n  Tools: {[tc['tool'] for tc in result['tool_calls']]} | Turns: {result['turns']}")
        return

    print("\n  Agent Chat — type 'quit' to exit | 'reset' to start over")
    history, session = [], None
    while True:
        try:
            q = input("\n  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q or q.lower() == "quit":
            break
        if q.lower() == "reset":
            history, session = [], None; print("  Conversation reset."); continue
        result  = agent_run(q, history, session_id=session, verbose=True)
        history = result["history"]
        session = result.get("session_id")


# ── finetune ──────────────────────────────────────────────────────────────────

def cmd_finetune(args):
    if args.subcommand == "prep":
        from finetuning.data_prep import run
        run(preview=args.preview)

    elif args.subcommand == "train":
        from finetuning.trainer import run
        run(dry_run=args.dry_run, resume=args.resume, merge=not args.no_merge)

    elif args.subcommand == "serve":
        from finetuning.trainer import serve
        serve(mode=args.mode, port=args.port)

    else:
        print("  Usage: python main.py finetune [prep|train|serve]")


# ── eval ──────────────────────────────────────────────────────────────────────

def cmd_eval(args):
    import config

    if args.list:
        if not os.path.exists(config.RESULTS_DIR):
            print("  No results yet."); return
        for f in sorted(os.listdir(config.RESULTS_DIR)):
            if not f.endswith(".json"): continue
            with open(os.path.join(config.RESULTS_DIR,f)) as fp:
                r = json.load(fp)
            s = r["summary"]
            g = "✅" if s["ci_gate"]=="PASS" else "❌"
            print(f"  {g} {f[:50]} | {s['pass_rate']:.0%}")
        return

    if args.compare:
        a_path, b_path = args.compare
        with open(a_path) as f: ra = json.load(f)
        with open(b_path) as f: rb = json.load(f)
        ra_res = {r["id"]:r for r in ra["results"]}
        rb_res = {r["id"]:r for r in rb["results"]}
        print(f"\n  {ra['model']} → {rb['model']}")
        print(f"  Pass rate: {ra['summary']['pass_rate']:.1%} → {rb['summary']['pass_rate']:.1%}")
        for tid in ra_res:
            if tid not in rb_res: continue
            d = round(rb_res[tid].get("overall_score",0) - ra_res[tid].get("overall_score",0), 2)
            if abs(d) > 0.05:
                icon = "✅" if d > 0 else "❌"
                print(f"  {icon} {tid}: {d:+.2f}")
        return

    if args.target:
        config.EVAL_TARGET = args.target

    from evaluation.eval_pipeline import run
    r = run(use_semantic=not args.no_semantic,
            use_judge=not args.no_judge,
            category=args.category)
    sys.exit(0 if r["summary"]["ci_gate"]=="PASS" else 1)


# ── CLI setup ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(prog="main.py", description="Future AI — unified pipeline CLI")
    sub = p.add_subparsers(dest="command")

    # status
    sub.add_parser("status", help="Show system status")

    # ingest
    pi = sub.add_parser("ingest", help="Build/refresh vector store")
    pi.add_argument("--source",      nargs="+", default=["local"], choices=["local","gitlab"])
    pi.add_argument("--reset",       action="store_true")
    pi.add_argument("--branch",      default="main")
    pi.add_argument("--extensions",  nargs="+")
    pi.add_argument("--no-issues",   action="store_true")
    pi.add_argument("--no-mrs",      action="store_true")
    pi.add_argument("--no-wiki",     action="store_true")
    pi.add_argument("--stats",       action="store_true")

    # chat
    pc = sub.add_parser("chat", help="RAG question answering")
    pc.add_argument("--question", "-q", default=None)
    pc.add_argument("--filter",         default=None, choices=["code","issue","wiki","document"])
    pc.add_argument("--debug",          action="store_true")

    # agent
    pa = sub.add_parser("agent", help="Agentic mode with tools")
    pa.add_argument("--demo",     action="store_true")
    pa.add_argument("--scenario", type=int, default=None)

    # finetune
    pf = sub.add_parser("finetune", help="Fine-tuning pipeline")
    pf.add_argument("subcommand",  choices=["prep","train","serve"])
    pf.add_argument("--preview",   action="store_true")
    pf.add_argument("--dry-run",   action="store_true")
    pf.add_argument("--resume",    action="store_true")
    pf.add_argument("--no-merge",  action="store_true")
    pf.add_argument("--mode",      default="merged", choices=["merged","lora"])
    pf.add_argument("--port",      type=int, default=8000)

    # eval
    pe = sub.add_parser("eval", help="Evaluation pipeline")
    pe.add_argument("--no-judge",    action="store_true")
    pe.add_argument("--no-semantic", action="store_true")
    pe.add_argument("--category",    default=None)
    pe.add_argument("--target",      default=None, choices=["openai","vllm","rag"])
    pe.add_argument("--list",        action="store_true")
    pe.add_argument("--compare",     nargs=2, metavar=("A","B"))

    args = p.parse_args()

    if   args.command == "status":   cmd_status()
    elif args.command == "ingest":   cmd_ingest(args)
    elif args.command == "chat":     cmd_chat(args)
    elif args.command == "agent":    cmd_agent(args)
    elif args.command == "finetune": cmd_finetune(args)
    elif args.command == "eval":     cmd_eval(args)
    else:
        p.print_help()

if __name__ == "__main__":
    main()

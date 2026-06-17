"""
generate.py
-----------
Reads seeds.py and generates a complete fine-tuning dataset.
This file never needs to be edited — edit seeds.py instead.

BACKENDS (--backend flag):
  openai   → GPT-4o via api.openai.com. Best quality. Needs API key + money (~$1-3 full run).
  ollama   → LLaMA/Mistral on your own machine. FREE. No API key.
             Install: https://ollama.com  then: 
             ollama pull llama3.2 
             ollama serve 
  vllm     → installed with "pip install vllm". FREE. No API key.
             start with "vllm serve mistralai/Mistral-7B-Instruct-v0.2"
             Must have: python main.py finetune serve running

Usage:
  python generate.py --all                        # use openai (default)
  python generate.py --all --backend ollama       # FREE, uses local Ollama
  python generate.py --all --backend vllm         # FREE, uses your vLLM server
  python generate.py --tone                       # expand tone examples only
  python generate.py --distillation               # answer domain questions
  python generate.py --structured                 # structured output examples
  python generate.py --reasoning                  # reasoning pattern examples
  python generate.py --refusals                   # safety boundary examples
  python generate.py --synthetic                  # topic-based synthetic examples
  python generate.py --gitlab                     # FREE — mine real examples from GitLab
  python generate.py --gitlab --gitlab-max-issues 200 --gitlab-max-mrs 100
  python generate.py --merge                      # combine all → training_examples.json
  python generate.py --stats                      # show what's been generated
  python generate.py --preview tone               # preview output before generating

  GITLAB REQUIRES: pip install python-gitlab
#            GITLAB_URL, GITLAB_TOKEN, GITLAB_PROJECT_ID in .env
"""

import json
import os
import argparse
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv() # Carica le variabili dal file .env nel sistema

import seeds
import promts


# ════════════════════════════════════════════════════════════════════════
#  BACKEND SETUP
#  The only difference between backends is base_url + model name.
#  Every API call below is identical regardless of backend.
# ════════════════════════════════════════════════════════════════════════

BACKENDS = {
    "openai": {
        "base_url":   None,                          # uses OpenAI default
        "api_key":    os.getenv("OPENAI_API_KEY",""),
        "model":      "gpt-4o",
        "note":       "GPT-4o via api.openai.com (~$1-3 for full run)",
    },
    "ollama": {
        "base_url":   "http://localhost:11434/v1",
        "api_key":    "ollama",                      # required but ignored by Ollama
        "model":      os.getenv("OLLAMA_MODEL","llama3.1"),
        "note":       "Local Ollama — FREE. Install: https://ollama.com then: ollama pull llama3.1",
    },
    "vllm": {
        "base_url":   os.getenv("VLLM_CHAT_URL","http://localhost:8000/v1"),
        "api_key":    "x",
        "model":      os.getenv("VLLM_CHAT_MODEL","finetuned-model"),
        "note":       "Your local vLLM server — FREE. Must be running first.",
    },
}

# Will be set by --backend flag in main()
client       = None
active_model = None

OUTPUT_DIR   = "data/generated"
FINAL_OUTPUT = "data/training_examples.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

class Fine_tuning

    def __init__(self,backend:str)

    def init_backend(backend_name: str):
        global client, active_model
        if backend_name not in BACKENDS:
            raise ValueError(f"Unknown backend: {backend_name}. Choose: {list(BACKENDS)}")

        #get all info of backend
        b = BACKENDS[backend_name]
        print(f"\n  Backend: {backend_name} — {b['note']}")

        #construct kwargs
        kwargs = {"api_key": b["api_key"]}
        if b["base_url"]:
            kwargs["base_url"] = b["base_url"]

        #call this in all cases, OpenAi is just a python function that makes HTTP requests in the OpenAI API format. 
        #vLLM and Ollama both deliberately implement the exact same API format as OpenAI
        client       = OpenAI(**kwargs)
        active_model = b["model"]

        # Quick connectivity check
        try:
            client.models.list()
            print(f"  Model:   {active_model} ✅\n")
        except Exception as e:
            print(f"  ⚠️  Could not connect: {e}")
            if backend_name == "ollama":
                print("     Make sure Ollama is running: ollama serve")
                print(f"    And model is pulled: ollama pull {active_model}")
            elif backend_name == "vllm":
                print("     Make sure vLLM is running: python main.py finetune serve")
            raise  #raise the original error so we won't loose it

    def generate(seed_samples: list[dict],
                source: str,
                prompt_init: str,
                count:int,
                extra_instructions: str='',
                temperature: float=0.8) -> list[dict]:

        '''generic function, takes seed examples, asks the LLM for more like them.
        Used by tone, reasoning, and refusals — all three do the same thing.
    
        seed_examples:       the hand-written examples from seeds.py
        count:               how many new examples to generate
        extra_instructions:  any extra guidance for this specific type'''

        #convert json text to array
        examples_text = json.dumps(seed_samples, indent=2) #get TOME_EXAMPLES in seeds.py file formatted and readable (indent=2)

        #construct prompt
        prompt = prompts_init.format(
            context=seeds.COMPANY_CONTEXT,
            examples=examples_text,
            count=count,
            extra_instructions = extra_instructions
        )

            #temp 0.8 says be creative and various
        raw      = call_gpt4o(prompt, temperature=temperature)

        #parse_list converts raw text in objects
        #validate each object to have the required structure
        examples = validate(parse_list(raw))

        #add metadata
        for ex in examples:
            ex["metadata"] = {"source": f"{source}_expansion", "generated": True}

        # Also include the original seed examples and add metadata
        seeds_tagged = [{**e, "metadata": {"source": f"{source}_seed", "generated": False}}
                        for e in seed_samples]

        #combine evrything
        all_examples = seeds_tagged + examples

        #save it to json file called like the source
        save_batch(all_examples, source)
        return all_examples




    def generate_tone(count: int = 50) -> list[dict]:
        return generate(
                        seed_examples=seeds.TONE_EXAMPLES,
                        source="tone",
                        prompt_init=prompts.TONE_EXPAND_PROMPT,
                        count=count
        )



    def generate_distillation() -> list[dict]:
        print(f"\n  Generating distillation examples for {len(seeds.DOMAIN_QUESTIONS)} questions...")
        examples = []

        #domain questions is an array not a json
        for i, question in enumerate(seeds.DOMAIN_QUESTIONS, 1):
            print(f"  [{i}/{len(seeds.DOMAIN_QUESTIONS)}] {question[:65]}...")

            # Distillation uses plain text response (not JSON format)
            # call standard py method to ask a question and receive an answer
            resp = client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", 
                    "content":   f"You are a senior backend engineer.\n{seeds.COMPANY_CONTEXT}"},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
            )
            answer = resp.choices[0].message.content.strip()
            examples.append({
                "instruction": question,
                "input":       "",
                "output":      answer,
                "metadata":    {"source": "distillation_gpt4o", "generated": True}
            })
        save_batch(examples, "distillation")
        return examples


    def generate_structured(count_per_schema: int = 20) -> list[dict]:
        print(f"\n  Generating structured output examples...")
        all_examples = []

        for schema_def in seeds.STRUCTURED_SCHEMAS:
            print(f"  Schema: {schema_def['name']} ({count_per_schema} examples)...")
            prompt = prompts.STRUCTURED_EXPAND_PROMPT.format(
                count=count_per_schema,
                instruction=schema_def["instruction"],
                schema=json.dumps(schema_def["schema"], indent=2),
                examples=json.dumps(schema_def["examples"], indent=2),
            )
            raw      = call_gpt4o(prompt, temperature=0.7)
            new_exs  = parse_list(raw)

            # Convert to full training format with the instruction prepended
            formatted = []
            # Include seed examples first
            for ex in schema_def["examples"]:
                formatted.append({
                    "instruction": schema_def["instruction"],
                    "input":       ex["input"],
                    "output":      ex["output"],
                    "metadata":    {"source": f"structured_{schema_def['name']}_seed", "generated": False}
                })
            # Then generated
            for ex in new_exs:
                if ex.get("input") and ex.get("output"):
                    formatted.append({
                        "instruction": schema_def["instruction"],
                        "input":       ex["input"],
                        "output":      ex["output"],
                        "metadata":    {"source": f"structured_{schema_def['name']}", "generated": True}
                    })

            all_examples.extend(formatted)
            print(f"    → {len(formatted)} examples for {schema_def['name']}")

        save_batch(all_examples, "structured")
        return all_examples



    def generate_reasoning(count: int = 20) -> list[dict]:
        return generate(
                        seed_examples=seeds.REASONING_EXAMPLES,
                        source="reasoning",
                        prompt_init=prompts.REASONING_EXPAND_PROMPT,
                        count=count
        )

    def generate_refusals(count: int = 20) -> list[dict]:
        return generate(
                        seed_examples=seeds.REFUSAL_EXAMPLES,
                        source="refusal",
                        prompt_init=prompts.REFUSAL_EXPAND_PROMPT,
                        count=count
        )



    def generate_synthetic(count_per_topic: int = 5) -> list[dict]:
        print(f"\n  Generating synthetic examples for {len(seeds.SYNTHETIC_TOPICS)} topics ({count_per_topic} each)...")
        all_examples = []
        style_example = json.dumps(seeds.TONE_EXAMPLES[0], indent=2)

        for i, topic in enumerate(seeds.SYNTHETIC_TOPICS, 1):
            print(f"  [{i}/{len(seeds.SYNTHETIC_TOPICS)}] {topic}...")
            prompt = prompts.SYNTHETIC_PROMPT.format(
                count=count_per_topic,
                topic=topic,
                context=seeds.COMPANY_CONTEXT,
                style_example=style_example,
            )
            raw      = call_gpt4o(prompt, temperature=0.8)
            examples = validate(parse_list(raw))
            for ex in examples:
                ex["metadata"] = {"source": "synthetic", "topic": topic, "generated": True}
            all_examples.extend(examples)

        save_batch(all_examples, "synthetic")
        return all_examples



# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════

def save_batch(examples: list[dict], name: str) -> str:
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    # Load existing if present (append, don't overwrite)

    existing = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)

    combined = existing + examples

    with open(path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"  Saved {len(examples)} new examples → {path} ({len(combined)} total)")
    return path

def call_gpt4o(prompt: str, temperature: float = 0.7,
               json_mode: bool = True) -> str:
    """Call whichever backend is active. Falls back gracefully if json_mode unsupported."""
    kwargs = dict(model=active_model,
                  messages=[{"role": "user", "content": prompt}],
                  temperature=temperature)
    if json_mode:
        try:
            resp = client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"})
        except Exception:
            resp = client.chat.completions.create(**kwargs)
    else:
        resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

def parse_list(raw: str) -> list[dict]:
    """Extract a list of dicts from GPT-4o JSON response."""
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return parsed
    # sometimes wrapped in a key like {"examples": [...]}
    for v in parsed.values():
        if isinstance(v, list):
            return v
    return []

def validate(examples: list[dict]) -> list[dict]:
    """Keep only examples that have instruction and output."""
    valid = []
    for ex in examples:
        if ex.get("instruction") and ex.get("output"):
            if not ex.get("input"):
                ex["input"] = ""
            valid.append(ex)
    return valid



# ════════════════════════════════════════════════════════════════════════
#  MERGE — combine all generated files into one training_examples.json
# ════════════════════════════════════════════════════════════════════════

def merge():
    """Combine all generated JSON files into the final training dataset."""
    all_examples = []
    sources      = {}

    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path) as f:
            examples = json.load(f)
        all_examples.extend(examples)
        sources[fname] = len(examples)

    with open(FINAL_OUTPUT, "w") as f:
        json.dump(all_examples, f, indent=2)

    print(f"\n  ── Merged training dataset ─────────────────────")
    for src, count in sources.items():
        print(f"  {src:<35} {count:>4} examples")
    print(f"  {'─'*42}")
    print(f"  {'TOTAL':<35} {len(all_examples):>4} examples")
    print(f"\n  Output → {FINAL_OUTPUT}")
    print(f"  Ready for: python main.py finetune prep\n")
    return all_examples


# ════════════════════════════════════════════════════════════════════════
#  STATS — show what's been generated so far
# ════════════════════════════════════════════════════════════════════════

def stats():
    print(f"\n  ── Generated data stats ────────────────────────")
    total = 0
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(OUTPUT_DIR, fname)) as f:
            n = len(json.load(f))
        total += n
        print(f"  {fname:<35} {n:>4} examples")
    print(f"  {'─'*42}")
    print(f"  {'TOTAL':<35} {total:>4} examples")

    if os.path.exists(FINAL_OUTPUT):
        with open(FINAL_OUTPUT) as f:
            merged_count = len(json.load(f))
        print(f"\n  Merged training_examples.json: {merged_count} examples")
    print()


# ════════════════════════════════════════════════════════════════════════
#  PREVIEW — show sample output before committing to a full generation
# ════════════════════════════════════════════════════════════════════════

def preview(type_name: str):
    """Show 2 examples of what will be generated, without saving."""
    print(f"\n  Preview: {type_name} (2 examples, not saved)\n")

    if type_name == "tone":
        result = generate_tone(count=2)
        for ex in result[:2]:
            _print_example(ex)

    elif type_name == "distillation":
        q = seeds.DOMAIN_QUESTIONS[0]
        resp = client.chat.completions.create(
            model=active_model,
            messages=[{"role": "system", "content": seeds.COMPANY_CONTEXT},
                      {"role": "user", "content": q}],
            temperature=0.3,
        )
        _print_example({"instruction": q, "input": "", "output": resp.choices[0].message.content[:300] + "..."})

    elif type_name == "structured":
        schema = seeds.STRUCTURED_SCHEMAS[0]
        print(f"  Schema: {schema['name']}")
        print(f"  Instruction: {schema['instruction']}")
        print(f"  Example input:  {schema['examples'][0]['input']}")
        print(f"  Example output: {schema['examples'][0]['output']}\n")

    elif type_name == "refusals":
        print(f"  First seed refusal:")
        _print_example(seeds.REFUSAL_EXAMPLES[0])

def _print_example(ex: dict):
    print(f"  instruction: {ex['instruction']}")
    if ex.get("input"):
        print(f"  input:       {ex['input'][:80]}")
    print(f"  output:      {ex['output'][:200]}...")
    print()


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════
#  GENERATOR 7 — GITLAB MINING
#  Extract real training examples from your GitLab issues and MRs.
#  Real conversations from your team = highest quality training data.
#
#  HOW IT WORKS:
#  Issues:  issue title + description = "instruction + input"
#           last resolution comment   = "output"
#  MRs:     MR title + description    = "instruction + input"
#           longest review comment    = "output"
#
#  REQUIRES: pip install python-gitlab
#            GITLAB_URL, GITLAB_TOKEN, GITLAB_PROJECT_ID in .env
#  IMPORTANT: review data/generated/gitlab.json before merging.
# ════════════════════════════════════════════════════════════════════════

def mine_gitlab(max_issues: int = 100, max_mrs: int = 50,
                min_output_length: int = 100) -> list[dict]:
    """
    Mine real training examples from GitLab issues and MRs.
    min_output_length: filters out short useless comments ("fixed", "LGTM", "+1").
    GitLab is not needed for generation — this runs independently of the backend.
    """
    try:
        import gitlab
    except ImportError:
        print("  Missing: pip install python-gitlab")
        return []

    gitlab_url   = os.getenv("GITLAB_URL", "https://gitlab.com")
    gitlab_token = os.getenv("GITLAB_TOKEN", "")
    project_id   = os.getenv("GITLAB_PROJECT_ID", "")

    if not gitlab_token:
        print("  GITLAB_TOKEN not set in .env"); return []
    if not project_id:
        print("  GITLAB_PROJECT_ID not set in .env"); return []

    gl      = gitlab.Gitlab(gitlab_url, private_token=gitlab_token)
    gl.auth()
    project = gl.projects.get(project_id)
    print(f"  Connected: {project.name_with_namespace}")

    examples = []

    # Issues
    print(f"  Mining closed issues (max {max_issues})...")
    issues = project.issues.list(state="closed", per_page=100, get_all=False)
    issue_count = 0
    for issue in issues[:max_issues]:
        if not issue.description or len(issue.description) < 30:
            continue
        notes      = issue.notes.list(get_all=True)
        user_notes = [n for n in notes
                      if not n.system and len(n.body) >= min_output_length]
        if not user_notes:
            continue
        resolution = user_notes[-1].body
        examples.append({
            "instruction": f"A developer reported this issue: {issue.title}",
            "input":       issue.description[:600].strip(),
            "output":      resolution[:1000].strip(),
            "metadata":    {"source":"gitlab_issue","issue_id":issue.iid,
                            "url":issue.web_url,"generated":False},
        })
        issue_count += 1
    print(f"    Issues: {issue_count}")

    # MRs
    print(f"  Mining merged MRs (max {max_mrs})...")
    mrs = project.mergerequests.list(state="merged", per_page=100, get_all=False)
    mr_count = 0
    for mr in mrs[:max_mrs]:
        if not mr.description or len(mr.description) < 80:
            continue
        notes      = mr.notes.list(get_all=True)
        user_notes = [n for n in notes
                      if not n.system and len(n.body) >= min_output_length]
        if not user_notes:
            continue
        best = max(user_notes, key=lambda n: len(n.body))
        examples.append({
            "instruction": f"Review this merge request: {mr.title}",
            "input":       mr.description[:600].strip(),
            "output":      best.body[:1000].strip(),
            "metadata":    {"source":"gitlab_mr","mr_id":mr.iid,
                            "url":mr.web_url,"generated":False},
        })
        mr_count += 1
    print(f"    MRs: {mr_count}")
    print(f"    Total: {len(examples)} GitLab examples")
    print(f"  Review data/generated/gitlab.json before merging.")

    save_batch(examples, "gitlab")
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fine-tuning training data from seeds.py")
    parser.add_argument("--backend", default="openai",
                        choices=["openai","ollama","vllm"],
                        help="openai=paid, ollama=free local, vllm=free local server")
    parser.add_argument("--all",          action="store_true", help="Run all generators")
    parser.add_argument("--tone",         action="store_true", help="Expand tone examples")
    parser.add_argument("--distillation", action="store_true", help="GPT-4o answers domain questions")
    parser.add_argument("--structured",   action="store_true", help="Generate structured output examples")
    parser.add_argument("--reasoning",    action="store_true", help="Expand reasoning examples")
    parser.add_argument("--refusals",     action="store_true", help="Expand refusal examples")
    parser.add_argument("--synthetic",    action="store_true", help="Generate topic-based synthetic examples")
    parser.add_argument("--gitlab",        action="store_true", help="Mine real examples from GitLab issues and MRs (no LLM needed)")
    parser.add_argument("--merge",        action="store_true", help="Merge all generated files")
    parser.add_argument("--stats",        action="store_true", help="Show stats")
    parser.add_argument("--preview",      default=None,
                        choices=["tone","distillation","structured","refusals"],
                        help="Preview output without saving")

    # Count controls
    parser.add_argument("--tone-count",     type=int, default=50)
    parser.add_argument("--reasoning-count", type=int, default=20)
    parser.add_argument("--refusal-count",  type=int, default=30)
    parser.add_argument("--structured-count", type=int, default=20)
    parser.add_argument("--synthetic-per-topic", type=int, default=5)
    parser.add_argument("--gitlab-max-issues", type=int, default=100)
    parser.add_argument("--gitlab-max-mrs",    type=int, default=50)

    args = parser.parse_args() #this line actually reads the arguments passed to the script and stores them in the args dict

    # Initialise the backend ONCE here — all generators use it
    needs_llm = any([args.all, args.tone, args.distillation, args.structured,
                     args.reasoning, args.refusals, args.synthetic,
                     args.preview])
   
    #if at least one is true init backend (only for openai/llama/vllm case)
    if needs_llm:
        init_backend(args.backend)

    if args.preview:
        preview(args.preview)

    elif args.stats:
        stats()

    elif args.merge:
        merge()

    else:
        ran_something = False

        if args.all or args.tone:
            generate_tone(count=args.tone_count)
            ran_something = True

        if args.all or args.distillation:
            generate_distillation()
            ran_something = True

        if args.all or args.structured:
            generate_structured(count_per_schema=args.structured_count)
            ran_something = True

        if args.all or args.reasoning:
            generate_reasoning(count=args.reasoning_count)
            ran_something = True

        if args.all or args.refusals:
            generate_refusals(count=args.refusal_count)
            ran_something = True

        if args.all or args.synthetic:
            generate_synthetic(count_per_topic=args.synthetic_per_topic)
            ran_something = True

        if args.all or args.gitlab:
            mine_gitlab(max_issues=args.gitlab_max_issues,
                        max_mrs=args.gitlab_max_mrs)
            ran_something = True

        if ran_something:
            print("\n  All done. Run --merge to combine into training_examples.json")
            stats()

        if not ran_something:
            parser.print_help()

# Fine-tuning Data Generator

Two files. One you edit, one you never touch.

```
seeds.py     ← EDIT THIS — your company context, examples, topics
generate.py  ← DON'T TOUCH — reads seeds.py and generates everything
data/
  generated/
    tone.json          ← expanded tone examples
    distillation.json  ← GPT-4o domain Q&A
    structured.json    ← JSON output format examples
    reasoning.json     ← step-by-step reasoning examples
    refusals.json      ← safety boundary examples
    synthetic.json     ← topic-based synthetic examples
  training_examples.json  ← final merged output (→ fine-tuning)
```

## Step 1: Edit seeds.py

Open `seeds.py` and change:
- `COMPANY_CONTEXT` — who you are, your stack, your tone
- `TONE_EXAMPLES` — 5-10 examples written in your exact voice
- `DOMAIN_QUESTIONS` — real questions your users actually ask
- `STRUCTURED_SCHEMAS` — JSON schemas you want reliable output for
- `REASONING_EXAMPLES` — how you want problems reasoned through
- `REFUSAL_EXAMPLES` — things the model must not do
- `SYNTHETIC_TOPICS` — topics relevant to your team

## Step 2: Preview before generating

```bash
# See what tone examples will look like (2 samples, not saved)
python generate.py --preview tone

# See a distillation example
python generate.py --preview distillation

# See a structured output example
python generate.py --preview structured
```

## Step 3: Generate

```bash
# Generate everything (costs ~$2-5 in OpenAI API calls for full run)
python generate.py --all

# Or generate specific types
python generate.py --distillation       # answer your domain questions
python generate.py --tone               # expand your tone examples
python generate.py --structured         # JSON output examples
python generate.py --reasoning          # step-by-step thinking examples
python generate.py --refusals           # safety boundaries
python generate.py --synthetic          # topic-based variety

# Control counts
python generate.py --tone --tone-count 100
python generate.py --synthetic --synthetic-per-topic 10
```

## Step 4: Check what was generated

```bash
python generate.py --stats
```

```
  tone.json                           55 examples
  distillation.json                   25 examples
  structured.json                     71 examples
  reasoning.json                      22 examples
  refusals.json                       36 examples
  synthetic.json                     100 examples
  ─────────────────────────────────────────────
  TOTAL                              309 examples
```

## Step 5: Review the output

Open the JSON files in `data/generated/` and read through them.
Delete any examples that are wrong, off-brand, or low quality.
Good training data is more important than quantity.

## Step 6: Merge and fine-tune

```bash
# Combine all generated files into training_examples.json
python generate.py --merge

# Then fine-tune
python main.py finetune prep
python main.py finetune train
```

## What each generator costs (approximate)

| Generator | Examples | GPT-4o calls | Approx cost |
|---|---|---|---|
| tone (50) | 50 | 3 | $0.10 |
| distillation (25 questions) | 25 | 25 | $0.50 |
| structured (3 schemas × 20) | 60 | 3 | $0.15 |
| reasoning (20) | 20 | 1 | $0.05 |
| refusals (30) | 30 | 1 | $0.05 |
| synthetic (20 topics × 5) | 100 | 20 | $0.40 |
| **TOTAL** | **285** | **~53** | **~$1.25** |

Full run costs about $1-3 depending on response lengths.

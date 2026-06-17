# Knowledge Assistant AI — Unified AI Pipeline

All five pipelines in one project, controlled from a single CLI.

```
knowledge_assistant/
├── main.py                     ← single entry point for everything
├── config.py                   ← all settings in one place
├── .env.example
├── requirements.txt
│
├── ingestion/                  ← pipeline 2: build vector store
│   ├── loaders/
│   │   ├── local_loader.py     ← .txt and .pdf files
│   │   └── gitlab_loader.py    ← GitLab: code, issues, MRs, wiki
│   ├── chunker.py              ← split docs in chunks
│   ├── embedder.py             ← OpenAI or vLLM embeddings to turn chunks into usable vectors
│   └── vector_store.py         ← ChromaDB/pgvector persistence
│
├── inference/                  ← pipeline 1: RAG
│   └── pipeline.py             ← retrieve + call LLM
│
├── finetuning/                 ← pipeline 3: LoRA fine-tuning
│   ├── data_prep.py            ← format training data
│   └── trainer.py              ← train + serve with vLLM
│
├── evaluation/                 ← pipeline 4: eval + CI gate
│   └── eval_pipeline.py        ← keyword + semantic + LLM judge
│
├── agents/                     ← pipeline 5: agentic loop
│   └── agent.py                ← tools + decision loop
│
└── data/
    ├── raw/                    ← put your .txt/.pdf docs here
    ├── formatted/              ← training data (auto-created)
    ├── chroma_db/              ← vector store (auto-created)
    ├── orders/                 ← orders.json for agent demos
    ├── test_cases/             ← test_cases.json for eval
    ├── results/                ← eval reports (auto-created)
    └── logs/                   ← agent session logs
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# add your OPENAI_API_KEY

# Check what's ready
python main.py status

# Ingest local docs
python main.py ingest --source local

# Ingest from GitLab
python main.py ingest --source gitlab

# Both
python main.py ingest --source local gitlab

# Chat with your docs
python main.py chat

# One question
python main.py chat -q "How does the payment service work?"

# shows which sources were retrieved + scores
python main.py chat --debug       

# only search code chunks (not wiki/issues) 
python main.py chat --filter code    

# one-shot, no interactive loop
python main.py chat -q "your question" 

# Agent mode (with tools)
python main.py agent

# Run demo scenarios
python main.py agent --demo

# Eval (fast, no LLM judge)
python main.py eval --no-judge

# Full eval
python main.py eval

# Fine-tune (needs GPU)
python main.py finetune prep
python main.py finetune train --dry-run
python main.py finetune train
python main.py finetune serve
```

## The Full Workflow

```
                ADD DOCUMENTS
                    │
         ┌──────────┴──────────┐
         │                     │
    data/raw/             GitLab repo
    (txt, pdf)      (code, issues, MRs, wiki)
         │                     │
         └──────────┬──────────┘
                    │
            python main.py ingest
                    │
              ChromaDB store
                    │
         ┌──────────┴──────────┐
         │                     │
  python main.py chat    python main.py agent
  (RAG answers)          (RAG + tools + orders)
         │
  python main.py eval
  (quality gate)
         │
  python main.py finetune train
  (improve model on your data)
         │
  python main.py finetune serve
  (vLLM serving your model)
         │
  set INFERENCE_BACKEND=vllm in .env
  python main.py chat    ← now uses YOUR model
```

## Switching Backends

Everything is controlled by .env — no code changes needed:

```bash
# Use OpenAI (default, easiest)
EMBEDDING_BACKEND=openai
INFERENCE_BACKEND=openai

# Use your fine-tuned model via vLLM
# (after: python main.py finetune serve)
INFERENCE_BACKEND=vllm
VLLM_CHAT_MODEL=finetuned-model

# Eval your RAG pipeline specifically
EVAL_TARGET=rag
python main.py eval
```

## Adding Your Own Data

1. Drop `.txt` or `.pdf` files into `data/raw/`
2. Run `python main.py ingest --source local --reset`
3. Run `python main.py chat` and ask questions about them

For GitLab, set the three env vars and run:
```bash
python main.py ingest --source gitlab --no-mrs   # skip MRs if too noisy
```

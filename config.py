"""
config.py — single source of truth for the entire project.

TWO MAIN SWITCHES you'll actually use day-to-day:

  MODEL_BACKEND      → which LLM runs your chat/agent
  EMBEDDING_BACKEND  → which model turns text into vectors

Everything else flows from these two choices.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# ════════════════════════════════════════════════════════════════════════
#  SWITCH 1 — CHAT / AGENT MODEL
#  Controls: inference pipeline, agent, fine-tuning base, vLLM serving
# ════════════════════════════════════════════════════════════════════════

# "openai"  → call api.openai.com (GPT-4o-mini). Easy, no GPU needed.
# "llama"   → serve Meta LLaMA 3.1 8B locally via vLLM. Best for technical Q&A.
# "mistral" → serve Mistral 7B locally via vLLM. Faster, good for chat/support.
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "openai")

# OpenAI settings (used when MODEL_BACKEND = "openai")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = "gpt-4o-mini"

# LLaMA settings (used when MODEL_BACKEND = "llama")
# After: python main.py finetune serve --model llama
LLAMA_BASE_MODEL    = "unsloth/Meta-Llama-3.1-8B-Instruct"  # base (before fine-tuning)
LLAMA_FINETUNED     = "finetuned-llama"                     # name when served via vLLM
LLAMA_VLLM_URL      = os.getenv("LLAMA_VLLM_URL", "http://localhost:8000/v1")

# Mistral settings (used when MODEL_BACKEND = "mistral")
# After: python main.py finetune serve --model mistral
MISTRAL_BASE_MODEL  = "unsloth/mistral-7b-instruct-v0.3"    # base (before fine-tuning)
MISTRAL_FINETUNED   = "finetuned-mistral"                   # name when served via vLLM
MISTRAL_VLLM_URL    = os.getenv("MISTRAL_VLLM_URL", "http://localhost:8001/v1")

# ════════════════════════════════════════════════════════════════════════
#  SWITCH 2 — EMBEDDING MODEL
#  Controls: ingestion (building the vector store) + inference (query search)
#
#  IMPORTANT: if you change this, you MUST re-run ingestion with --reset
#  because the vector dimensions change between models.
#  OpenAI text-embedding-3-small → 1536 dimensions
#  BAAI/bge-large-en-v1.5        → 1024 dimensions
#  You cannot mix embeddings from different models in the same store.
# ════════════════════════════════════════════════════════════════════════

# variables used in config.get_embed_client()
# "openai" → call api.openai.com for embeddings (~$0.02/million tokens)
# "vllm"  → serve BAAI/bge-large-en-v1.5 via vLLM, zero cost, stays on machine
#             Start with: vllm serve BAAI/bge-large-en-v1.5 --task embedding --port 8002
# "local" -> serve ollama
EMBEDDING_BACKEND  = os.getenv("EMBEDDING_BACKEND", "openai")
OPENAI_EMBED_MODEL = "text-embedding-3-small"        # 1536 dims

LOCAL_EMBED_MODEL  = os.getenv("LOCAL_EMBED_MODEL", "nomic-embed-text")
LOCAL_EMBED_URL    = os.getenv("LOCAL_EMBED_URL", "http://localhost:11434/v1")

VLLM_EMBED_URL="http://localhost:8001/v1"
VLLM_CHAT_URL="http://localhost:8000/v1"
VLLM_CHAT_MODEL="finetuned-model"

# ════════════════════════════════════════════════════════════════════════
#  SWITCH 3 — EVALUATION JUDGE
#  Controls: which model scores answers in the eval pipeline
#
#  "openai" → GPT-4o judges answers. Most reliable. Costs ~$0.001/test case.
#  "local"  → LLaMA 3.1 70B judges answers. Free. Slightly less reliable.
#             (uses the same vLLM URL as MODEL_BACKEND=llama)
# ════════════════════════════════════════════════════════════════════════

JUDGE_BACKEND  = os.getenv("JUDGE_BACKEND", "openai")
OPENAI_JUDGE_MODEL = "gpt-4o"                        # strongest OpenAI judge
LOCAL_JUDGE_MODEL  = "meta-llama/Meta-Llama-3.1-70B-Instruct"  # best local judge
# NOTE: 70B needs ~40GB VRAM (2x A40 or 1x A100). For single 24GB GPU use 8B judge:
# LOCAL_JUDGE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # weaker but fits anywhere

# ════════════════════════════════════════════════════════════════════════
#  VECTOR DATABASE — ChromaDB
#
#  ChromaDB is a vector database. It stores your embedded chunks and
#  lets you search them efficiently.
#
#  Compared to a plain JSON file (like in the tutorial intro):
#    JSON file:   loads all chunks into RAM, linear scan = slow at scale
#    ChromaDB:    indexed (HNSW algorithm), only loads what's needed,
#                 persists to disk at CHROMA_PATH, handles 1M+ chunks fine
#
#  The data lives at CHROMA_PATH on disk. Delete this folder to wipe the store.
#  Collections are like tables — we use one per project (CHROMA_COLLECTION).
# ════════════════════════════════════════════════════════════════════════

# "chroma"   → ChromaDB on disk (default, zero setup)
# "pgvector" → PostgreSQL + pgvector extension
#              Requires: #pip install psycopg2-binary pgvector
#              and: "CREATE EXTENSION IF NOT EXISTS vector;" in your DB
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")

# ChromaDB settings (used when VECTOR_STORE=chroma)
CHROMA_PATH       = "data/chroma_db"
CHROMA_COLLECTION = "acme_knowledge"

# pgvector settings (used when VECTOR_STORE=pgvector)
PGVECTOR_URL = os.getenv("PGVECTOR_URL", "postgresql://localhost:5432/knowledge_assistant")

# ════════════════════════════════════════════════════════════════════════
#  EVERYTHING ELSE
# ════════════════════════════════════════════════════════════════════════

# GitLab
GITLAB_URL        = os.getenv("GITLAB_URL", "https://gitlab.com")
GITLAB_TOKEN      = os.getenv("GITLAB_TOKEN", "")
GITLAB_PROJECT_ID = os.getenv("GITLAB_PROJECT_ID", "")

# Retrieval
TOP_K = 5

# Chunking
CHUNK_STRATEGY = "recursive"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 80

# Fine-tuning
LORA_RANK      = 16
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
NUM_EPOCHS     = 3
BATCH_SIZE     = 4
GRAD_ACCUM     = 4
LEARNING_RATE  = 2e-4
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT   = True

# Evaluation
EVAL_TARGET    = os.getenv("EVAL_TARGET", "rag")
PASS_THRESHOLD = 0.70

# Agent
AGENT_MAX_TURNS = 10
AGENT_TEMP      = 0.3

# Paths
RAW_DOCS_DIR        = "data/raw"
FORMATTED_DATA_PATH = "data/formatted/training_data.jsonl"
ORDERS_PATH         = "data/orders/orders.json"
TICKETS_PATH        = "data/orders/tickets.json"
TEST_CASES_PATH     = "data/test_cases/test_cases.json"
RESULTS_DIR         = "data/results"
LOGS_DIR            = "data/logs"
FINAL_MODEL_DIR     = "data/finetuned/lora_adapters"
MERGED_MODEL_DIR    = "data/finetuned/merged_model"

# Identity
PROJECT_NAME = "Knowledge Assistant"
COMPANY_NAME = "Future Corp"

# System prompts
SUPPORT_SYSTEM_PROMPT = f"""You are a helpful AI assistant for Future Corp.
Answer questions using only the provided context.
If you don't know, say so clearly. Never invent information."""

AGENT_SYSTEM_PROMPT = """You are an AI support agent for Future Corp.
You have tools to look up orders, search documentation, and create tickets.
Always use tools to get real data. Never guess."""

FINETUNING_SYSTEM_PROMPT = """You are a professional customer support agent for Future Corp.
Be concise, empathetic, and solution-focused. Always offer a next step."""


# ════════════════════════════════════════════════════════════════════════
#  HELPER: resolve active model settings
#  Call these instead of reading MODEL_BACKEND directly.
# ════════════════════════════════════════════════════════════════════════

def get_chat_client():
    """Return (openai_client, model_name) for the active MODEL_BACKEND."""
    from openai import OpenAI
    if MODEL_BACKEND == "openai":
        return OpenAI(api_key=OPENAI_API_KEY), OPENAI_CHAT_MODEL
    elif MODEL_BACKEND == "llama":
        return OpenAI(base_url=LLAMA_VLLM_URL, api_key="x"), LLAMA_FINETUNED
    elif MODEL_BACKEND == "mistral":
        return OpenAI(base_url=MISTRAL_VLLM_URL, api_key="x"), MISTRAL_FINETUNED
    else:
        raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND}")

def get_embed_client():
    from openai import OpenAI
    if EMBEDDING_BACKEND == "openai":
        return OpenAI(api_key=OPENAI_API_KEY), OPENAI_EMBED_MODEL
    elif EMBEDDING_BACKEND == "ollama":
        return OpenAI(base_url=LOCAL_EMBED_URL, api_key="x"), LOCAL_EMBED_MODEL
    elif EMBEDDING_BACKEND == "vllm":
        return OpenAI(base_url=VLLM_EMBED_URL, api_key="x"), VLLM_EMBED_MODEL
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: '{EMBEDDING_BACKEND}'. "
                         f"Choose: openai, local, ollama, vllm")

def get_judge_client():
    """Return (openai_client, model_name) for the active JUDGE_BACKEND."""
    from openai import OpenAI
    if JUDGE_BACKEND == "openai":
        return OpenAI(api_key=OPENAI_API_KEY), OPENAI_JUDGE_MODEL
    elif JUDGE_BACKEND == "local":
        # reuse the LLaMA vLLM server — just point at a bigger model
        return OpenAI(base_url=LLAMA_VLLM_URL, api_key="x"), LOCAL_JUDGE_MODEL
    else:
        raise ValueError(f"Unknown JUDGE_BACKEND: {JUDGE_BACKEND}")

def active_base_model() -> str:
    """Which base model to fine-tune."""
    return LLAMA_BASE_MODEL if MODEL_BACKEND in ("llama","openai") else MISTRAL_BASE_MODEL
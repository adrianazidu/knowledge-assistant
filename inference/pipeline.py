"""
inference/pipeline.py — RAG inference pipeline.

Retrieves from the vector store built by ingestion/ingest.py,
then calls the configured model (OpenAI or vLLM).

Usage:
  from inference.pipeline import ask
  result = ask("How does the payment service handle retries?")
  print(result["answer"])
  print(result["sources"])
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI
from ingestion import embedder, vector_store
import config


def _chat_client():
    if config.MODEL_BACKEND == "openai":
        return OpenAI(api_key=config.OPENAI_API_KEY), config.OPENAI_CHAT_MODEL
    return OpenAI(base_url=config.VLLM_CHAT_URL, api_key="x"), config.VLLM_CHAT_MODEL


def ask(question: str,
        top_k: int = None,
        type_filter: str = None,
        stream: bool = False) -> dict:
    """
    Full RAG pipeline for one question.

    type_filter: only retrieve from specific source types
      e.g. "code" | "issue" | "wiki" | "document" | None (all)

    Returns:
      answer   — the LLM's response
      sources  — chunks retrieved (for transparency / citations)
      prompt   — full prompt sent to LLM (for debugging)
    """
    # 1. Embed question
    q_embedding = embedder.embed_query(question)

    # 2. Retrieve
    chunks = vector_store.query(q_embedding,
                                top_k=top_k or config.TOP_K,
                                type_filter=type_filter)
    if not chunks:
        return {"answer": "I couldn't find relevant information to answer that.",
                "sources": [], "prompt": ""}

    # 3. Build prompt
    context = "\n\n".join(
        f"[Source: {c['metadata']['source']} | score: {c['score']:.2f}]\n{c['text']}"
        for c in chunks
    )
    prompt = (
        f"{config.SUPPORT_SYSTEM_PROMPT}\n\n"
        f"--- CONTEXT ---\n{context}\n--- END ---\n\n"
        f"Question: {question}\nAnswer:"
    )

    # 4. Call LLM
    client, model = _chat_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=stream,
    )

    if stream:
        # caller handles the stream object
        return {"stream": response, "sources": chunks}

    answer = response.choices[0].message.content.strip()
    return {
        "answer":  answer,
        "sources": [{"source": c["metadata"]["source"],
                     "score":  c["score"],
                     "text":   c["text"][:120] + "..."} for c in chunks],
        "prompt":  prompt,
    }

"""ingestion/embedder.py — embeds chunks via OpenAI or vLLM or Ollama. Transform chunks into vectors to be saved into chroma db or pgvector"""
import time
from openai import OpenAI
import config  # i need the functions inside config.py

def embed_chunks(chunks: list[dict], batch_size: int = 100) -> list[dict]:
    """used in ingestion.ingest.py to save to local database"""
    client, model = config.get_embed_client()
    total = len(chunks)
    print(f"  Backend: {config.EMBEDDING_BACKEND} ({model})")
    for start in range(0, total, batch_size):
        batch  = chunks[start:start + batch_size]
        resp   = client.embeddings.create(input=[c["text"] for c in batch], model=model)
        for i, chunk in enumerate(batch):
            chunk["embedding"] = resp.data[i].embedding
        print(f"  Embedded {min(start+batch_size, total)}/{total}", end="\r")
        if config.EMBEDDING_BACKEND == "openai" and start + batch_size < total:
            time.sleep(0.1)
    print()
    return chunks

def embed_query(text: str) -> list[float]:
    """used in inference.pipeline to ask questions"""
    client, model = config.get_embed_client()
    resp = client.embeddings.create(input=text, model=model)
    return resp.data[0].embedding

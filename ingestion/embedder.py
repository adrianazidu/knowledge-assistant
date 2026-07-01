"""ingestion/embedder.py — embeds chunks via OpenAI or vLLM or Ollama. Transform chunks into vectors to be saved into chroma db or pgvector"""
import time
from openai import OpenAI
import config  # i need the functions inside config.py
from ingestion import vector_store # connect to vector store

"""
to make sure we do not write same things and to make sure we always have the correct version saved
we should make a a scheduled night job that 
-Wipes everything, re-ingest from scratch
or
-  resets on frequently changing sources (issues,mr)
    # delete all chunks for this source
    col.delete(where={"source": "gitlab://issues/42"})
    # then embed and insert fresh
or
- content hash comparison - Hash-based skip is the most common real approach. Airflow, Prefect, and most ML pipeline frameworks have this pattern built in 
— they call it "incremental processing" or "change detection."
 You store a fingerprint of the source document and only reprocess when it changes.
or
- Event-driven ingestion - Instead of polling on a schedule, GitLab sends a webhook when an issue or file changes,
 and the ingestion pipeline processes only that specific document immediately. 
 requires for the infrastructuire to receive and queue webhooks
 or
 - time based - store last_ingested_at per source, and on each run only fetch GitLab items updated after that timestamp. 
 GitLab's API supports updated_after parameter on issues and MRs specifically for this purpose:
 project.issues.list(updated_after="2025-01-01T00:00:00Z")
"""


def get_existing_ids() -> set:
    """Return all chunk IDs currently in the store."""
    t0  = time.time()
    ids = vector_store.get_all_ids()
    print(f"  [TIMING] get_existing_ids: {time.time()-t0:.2f}s for {len(ids)} IDs")
    return ids

#deprecated - if chunk contents change, and their index not, they will not be updated in the databse
def embed_chunks_incremental_old(chunks: list[dict]) -> list[dict]:
    """fragile method - If content changes but the ID stays the same, you serve stale data"""
    existing_ids = get_existing_ids()

    new_chunks     = []
    skipped        = 0

    for chunk in chunks:
        chunk_id = f"{chunk['source']}::chunk_{chunk['chunk_index']}"
        if chunk_id in existing_ids:
            skipped += 1
            print("chunk present - SKIP",flush=True)
        else:
            new_chunks.append(chunk)
            print("chunk not present - ADD",flush=True)

    print(f"  Skipping {skipped} unchanged chunks",flush=True)
    print(f"  Embedding {len(new_chunks)} new chunks",flush=True)

    #if all the chinks hash already exist oin database return empty 
    if not new_chunks:
        return []

    return new_chunks

def embed_chunks_incremental(chunks: list[dict]) -> list[dict]:
    """function that uses content hash to make sure i update changed chunks"""
    import hashlib
    col = vector_store._chroma_col()
    existing = col.get(include=["metadatas"])

    #save a dictionary of data with key chunk_index_id and value content_hash
    existing_hashes = vector_store.get_all_hashes()

    new_chunks = []
    for chunk in chunks:

        #generate an unique id for each
        chunk_id = f"{chunk['source']}::chunk_{chunk['chunk_index']}"

        #compute unique hash code for every chunk
        content_hash = hashlib.sha256(chunk["text"].encode()).hexdigest()

        #check content of old chunk with the same id
        if existing_hashes.get(chunk_id) == content_hash:
            #print("chunk present - SKIP",flush=True)
            continue  # unchanged — true skip

        #print("chunk not present - ADD",flush=True)

        chunk["metadata"]["content_hash"] = content_hash
        new_chunks.append(chunk)
    return new_chunks

def embed_chunks(chunks: list[dict], batch_size: int = 100) -> list[dict]:
    """used in ingestion.ingest.py to embed chunks before saving to local database"""

    #filter alreay saved chunks (previously saved documents)
    chunks = embed_chunks_incremental(chunks)

    client, model = config.get_embed_client()
    total = len(chunks)
    print(f"  Backend: {config.EMBEDDING_BACKEND} ({model})",flush=True)

    for start in range(0, total, batch_size):
        batch  = chunks[start:start + batch_size]
        resp   = client.embeddings.create(input=[c["text"] for c in batch], model=model)
        for i, chunk in enumerate(batch):
            chunk["embedding"] = resp.data[i].embedding
        print(f"  Embedded {min(start+batch_size, total)}/{total}", end="\r",flush=True)
        if config.EMBEDDING_BACKEND == "openai" and start + batch_size < total:
            time.sleep(0.1)
    print()
    return chunks

def embed_query(text: str) -> list[float]:
    """used in inference.pipeline to ask questions"""
    client, model = config.get_embed_client()
    resp = client.embeddings.create(input=text, model=model)
    return resp.data[0].embedding

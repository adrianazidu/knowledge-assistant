"""
ingestion/vector_store.py
--------------------------
Abstraction over two vector store backends:

  VECTOR_STORE=chroma   → ChromaDB, persists to disk at CHROMA_PATH.
                          Zero setup. Good for local dev and small-medium scale.
                          Files live in data/chroma_db/ — delete to wipe.

  VECTOR_STORE=pgvector → PostgreSQL with pgvector extension.
                          Needs a running Postgres instance.
                          Better for production: survives server restarts,
                          works with your existing DB infrastructure,
                          supports SQL queries alongside vector search.

  Setup for pgvector:
    1. Install extension in Postgres:
         CREATE EXTENSION IF NOT EXISTS vector;
    2. Set in .env:
         VECTOR_STORE=pgvector
         PGVECTOR_URL=postgresql://user:pass@localhost:5432/yourdb
    3. Run ingestion — table is created automatically on first run.

The public API is identical for both backends:
  save(chunks)
  query(embedding, top_k, type_filter)
  stats()
  reset()

Switch backends by changing VECTOR_STORE in .env.
⚠️  If you switch backends, re-run ingestion from scratch —
    the two stores are independent and don't share data.
"""
import config


# ─────────────────────────────────────────────────────────────────────────────
#  CHROMADB BACKEND
# ─────────────────────────────────────────────────────────────────────────────

def _chroma_col():
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=config.CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

def _chroma_save(chunks: list[dict]):
    col = _chroma_col()
    col.upsert(
        ids        = [f"{c['source']}::chunk_{c['chunk_index']}" for c in chunks],
        embeddings = [c["embedding"] for c in chunks],
        documents  = [c["text"] for c in chunks],
        metadatas  = [{
            "source":         c["source"],
            "chunk_index":    c["chunk_index"],
            "type":           c["metadata"].get("type", "document"),
            "chunk_strategy": c["metadata"].get("chunk_strategy", ""),
            "chunk_size":     c["metadata"].get("chunk_size", 0),
        } for c in chunks],
    )
    print(f"  [chroma] Saved {len(chunks)} chunks → total: {col.count()}")

def _chroma_query(embedding, top_k, type_filter):
    col   = _chroma_col()
    where = {"type": type_filter} if type_filter else None
    res   = col.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text":     res["documents"][0][i],
         "metadata": res["metadatas"][0][i],
         "score":    1 - res["distances"][0][i]}
        for i in range(len(res["ids"][0]))
    ]

def _chroma_stats():
    col   = _chroma_col()
    count = col.count()
    if count == 0:
        return {"total": 0, "sources": [], "by_type": {}}
    sample = col.get(limit=count, include=["metadatas"])
    types  = {}
    for m in sample["metadatas"]:
        t = m.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    sources = list({m["source"] for m in sample["metadatas"]})
    return {"total": count, "sources": sorted(sources), "by_type": types}

def _chroma_reset():
    import shutil, os
    if os.path.exists(config.CHROMA_PATH):
        shutil.rmtree(config.CHROMA_PATH)
        print(f"  [chroma] Wiped {config.CHROMA_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
#  PGVECTOR BACKEND
#
#  Stores vectors in a Postgres table called `embeddings`.
#  Uses cosine similarity (<=> operator from pgvector).
#
#  The table schema:
#    id         TEXT PRIMARY KEY   — same stable ID as ChromaDB uses
#    source     TEXT               — filename or gitlab URL
#    chunk_index INT               — position within source document
#    type       TEXT               — code / issue / wiki / document
#    text       TEXT               — the actual chunk content
#    metadata   JSONB              — all other metadata as JSON
#    embedding  vector(N)          — the embedding vector
#                                    N must match your embedding model:
#                                    text-embedding-3-small → 1536
#                                    BAAI/bge-large-en-v1.5 → 1024
# ─────────────────────────────────────────────────────────────────────────────

def _pg_conn():
    """
    Returns a psycopg2 connection.
    Requires: pip install psycopg2-binary pgvector
    """
    try:
        import psycopg2
    except ImportError:
        raise ImportError("Run: pip install psycopg2-binary pgvector")

    return psycopg2.connect(config.PGVECTOR_URL)

def _pg_ensure_table(conn, dim: int):
    """
    Create the embeddings table if it doesn't exist.
    dim must match the embedding model's output dimension.
    Called automatically on first save.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id          TEXT PRIMARY KEY,
                source      TEXT,
                chunk_index INT,
                type        TEXT,
                text        TEXT,
                metadata    JSONB,
                embedding   vector({dim})
            );
        """)
        # Index for fast cosine similarity search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS embeddings_cosine_idx
            ON embeddings
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
    conn.commit()

def _pg_save(chunks: list[dict]):
    import json as _json
    conn = _pg_conn()
    dim  = len(chunks[0]["embedding"])
    _pg_ensure_table(conn, dim)

    with conn.cursor() as cur:
        for c in chunks:
            chunk_id = f"{c['source']}::chunk_{c['chunk_index']}"
            meta     = {k: v for k, v in c["metadata"].items()
                        if k != "embedding"}   # don't double-store the vector
            cur.execute("""
                INSERT INTO embeddings
                    (id, source, chunk_index, type, text, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    text      = EXCLUDED.text,
                    metadata  = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
            """, (
                chunk_id,
                c["source"],
                c["chunk_index"],
                c["metadata"].get("type", "document"),
                c["text"],
                _json.dumps(meta),
                c["embedding"],         # psycopg2 + pgvector handles list → vector
            ))
    conn.commit()
    conn.close()

    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings;")
            total = cur.fetchone()[0]
    print(f"  [pgvector] Saved {len(chunks)} chunks → total: {total}")

def _pg_query(embedding: list[float], top_k: int,
              type_filter: str = None) -> list[dict]:
    import json as _json
    conn = _pg_conn()

    # <=> is the pgvector cosine distance operator
    # 1 - distance = similarity
    where = "WHERE type = %s" if type_filter else ""
    params = [embedding, top_k]
    if type_filter:
        params = [embedding, type_filter, top_k]

    sql = f"""
        SELECT text, source, chunk_index, type, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM embeddings
        {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    # embedding appears twice — once for score, once for ordering
    if type_filter:
        params = [embedding, type_filter, embedding, top_k]
    else:
        params = [embedding, embedding, top_k]

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()

    return [
        {
            "text":     row[0],
            "metadata": {"source": row[1], "chunk_index": row[2],
                         "type": row[3], **_json.loads(row[4])},
            "score":    float(row[5]),
        }
        for row in rows
    ]

def _pg_stats() -> dict:
    import json as _json
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM embeddings;")
        total = cur.fetchone()[0]
        if total == 0:
            conn.close()
            return {"total": 0, "sources": [], "by_type": {}}
        cur.execute("SELECT type, COUNT(*) FROM embeddings GROUP BY type;")
        by_type = dict(cur.fetchall())
        cur.execute("SELECT DISTINCT source FROM embeddings;")
        sources = sorted([r[0] for r in cur.fetchall()])
    conn.close()
    return {"total": total, "sources": sources, "by_type": by_type}

def _pg_reset():
    conn = _pg_conn()
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS embeddings;")
    conn.commit()
    conn.close()
    print("  [pgvector] Dropped embeddings table")


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — identical regardless of backend
#  All callers use these four functions. Backend is invisible to them.
# ─────────────────────────────────────────────────────────────────────────────

def _backend():
    return config.VECTOR_STORE.lower()

def save(chunks: list[dict]):
    if _backend() == "pgvector":
        _pg_save(chunks)
    else:
        _chroma_save(chunks)

def query(embedding: list[float], top_k: int = None,
          type_filter: str = None) -> list[dict]:
    """function called also in inference pipeline to query the saved vectors"""
    if _backend() == "pgvector":
        return _pg_query(embedding, top_k or config.TOP_K, type_filter)
    else:
        return _chroma_query(embedding, top_k or config.TOP_K, type_filter)

def stats() -> dict:
    if _backend() == "pgvector":
        return _pg_stats()
    else:
        return _chroma_stats()

def reset():
    if _backend() == "pgvector":
        _pg_reset()
    else:
        _chroma_reset()
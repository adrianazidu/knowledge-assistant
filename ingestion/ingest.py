"""
ingestion/ingest.py — run this to build or refresh the vector store.

Your docs (txt/pdf/GitLab)
           │
     Step 1: Load      — reads files from data/raw/ or pulls from GitLab API
           │
     Step 2: Chunk     — splits each doc into small overlapping pieces (~500 chars)
           │              Code files are split at function/class boundaries
           │              Short issues/MRs are kept whole
           │
     Step 3: Embed     — sends each chunk to OpenAI (or local vLLM) to get a
           │              vector (list of ~1536 numbers) representing its meaning
           │
     Step 4: Save      — stores all vectors + text in ChromaDB on disk
                          (at data/chroma_db/)

So that when you ask a question later, the system can convert your question into a vector too, then find the chunks with the most similar vectors — that's retrieval-augmented generation (RAG). The LLM only sees    
  those relevant chunks, not the whole document.

  Key things to know:
  - If you change the embedding model, you must re-run with --reset (vector dimensions change)
  - Source can be local (files in data/raw/), gitlab, or both
  - The result persists to disk — you only need to re-run when your docs change

Sources (pick any combination):
  --source local   → data/raw/*.txt and *.pdf
  --source gitlab  → GitLab repo: code, issues, MRs, wiki

Usage:
  python -m ingestion.ingest --source local
  python -m ingestion.ingest --source gitlab
  python -m ingestion.ingest --source local gitlab       # both
  python -m ingestion.ingest --source gitlab --reset     # wipe and rebuild
  python -m ingestion.ingest --source gitlab --no-issues --no-mrs
  python -m ingestion.ingest --stats
"""
import config  #make sure i get the encoding configuration first before any print
import argparse, time, sys, os

#modify search modules path to parent folder of this one, 0 means max priority for this specific folder
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ingestion.loaders import local_loader  #import file from folder
from ingestion.loaders.gitlab_loader import GitlabLoader #import class inside file from folder
from ingestion import chunker, embedder, vector_store

from dotenv import load_dotenv
load_dotenv() # Carica le variabili dal file .env nel sistema

def run(sources: list[str],
        reset: bool = False,
        gitlab_branch: str = "master",
        gitlab_extensions: list = None,
        include_issues: bool = True,
        include_mrs: bool = True,
        include_wiki: bool = True):

    t0 = time.time()
    print("\n╔══════════════════════════════════════╗")
    print("║         INGESTION PIPELINE            ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Sources: {', '.join(sources)}")

    if reset:
        vector_store.reset()

    # Step 1: Load
    print("\n── Step 1: Load ────────────────────────")
    docs = []

    if "local" in sources:
        print("  Loading local files...")
        docs += local_loader.load( os.getenv("LOCAL_DIR_RAW",""))

    if "gitlab" in sources:
        print("  Loading from GitLab...")
        gitlab_loader = GitlabLoader(branch=gitlab_branch,
                                    extensions=gitlab_extensions,
                                    include_issues=include_issues,
                                    include_mrs=include_mrs,
                                    include_wiki=include_wiki,)
        docs += gitlab_loader.load()
    if not docs:
        print("  No documents found."); return
    print(f"  Total documents: {len(docs)}")

    # Step 2: Chunk
    print("\n── Step 2: Chunk ───────────────────────")
    chunks = []
    for doc in docs:
        chunks.extend(chunker.chunk(doc))
    print(f"  Total chunks: {len(chunks)}")

    # Step 3: Embed
    print("\n── Step 3: Embed ───────────────────────")
    chunks = embedder.embed_chunks(chunks)

    # Step 4: Save
    print("\n── Step 4: Save ────────────────────────")
    if chunks:
        vector_store.save(chunks)
    else:
        print("Nothing new to save")

    s = vector_store.stats()
    print(f"\n✅ Done in {time.time()-t0:.1f}s — {s['total']} chunks in store")
    for t, n in s.get("by_type", {}).items():
        print(f"   {t:<20} {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",      nargs="+", default=["local"],
                        choices=["local","gitlab"])
    parser.add_argument("--reset",       action="store_true")
    parser.add_argument("--branch",      default="master")
    parser.add_argument("--extensions",  nargs="+", default=None)
    parser.add_argument("--no-issues",   action="store_true")
    parser.add_argument("--no-mrs",      action="store_true")
    parser.add_argument("--no-wiki",     action="store_true")
    parser.add_argument("--stats",       action="store_true")
    a = parser.parse_args()

    if a.stats:
        s = vector_store.stats()
        print(f"\n  Chunks: {s['total']} | Types: {s.get('by_type',{})}\n")
    else:
        try:
            main_t0 = time.time()
            run(sources=a.source,
                reset=a.reset,
                gitlab_branch=a.branch,
                gitlab_extensions=a.extensions, 
                include_issues=not a.no_issues,
                include_mrs=not a.no_mrs,
                include_wiki=not a.no_wiki)
            print(f"INGESTION_STATUS: SUCCESS")
        except Exception as e:
            print(f"INGESTION_STATUS: FAILED after {time.time()-main_t0:.1f}s — {e}")
            raise

"""launch in bg with
Start-Process -NoNewWindow -RedirectStandardOutput "ingest_log.txt" -RedirectStandardError "ingest_error_log.txt" python -ArgumentList "-m", "ingestion.ingest", "--source", "local"
or
start /B python -m ingestion.ingest --source local > ingest_log.txt 2>&1
and monitor the file log
powershell Get-Content ingest_log.txt -Wait -Tail 20
windows equivalent of tail -f
"""
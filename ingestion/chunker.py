"""ingestion/chunker.py — splits documents into chunks."""
import re
import config

def chunk(doc: dict) -> list[dict]:
    """Route to the right strategy based on document type."""
    doc_type = doc["metadata"].get("type", "document")
    ext      = doc["metadata"].get("extension", "")

    # Code: split at function/class boundaries
    if doc_type == "code" and ext in (".py", ".ts", ".js", ".go"):
        chunks = _chunk_code(doc)
        if chunks:
            return chunks

    # Short issues/MRs: keep whole
    if doc_type in ("issue", "merge_request") and len(doc["text"]) < config.CHUNK_SIZE * 2:
        return [_make_chunk(doc, doc["text"], 0, "whole")]

    # Default: recursive
    return _chunk_recursive(doc)


def _chunk_code(doc: dict) -> list[dict]:
    ext     = doc["metadata"].get("extension", "")
    pattern = {
        ".py": r'\n(?=(?:def |class |async def ))',
        ".ts": r'\n(?=(?:function |class |const |export |async function ))',
        ".js": r'\n(?=(?:function |class |const |export |async function ))',
    }.get(ext)
    if not pattern:
        return []
    parts  = [p.strip() for p in re.split(pattern, doc["text"]) if p.strip()]
    result = []
    for i, part in enumerate(parts):
        if len(part) > config.CHUNK_SIZE * 2:
            sub = {**doc, "text": part}
            result.extend(_chunk_recursive(sub))
        else:
            result.append(_make_chunk(doc, part, i, "code_boundary"))
    return result


def _chunk_recursive(doc: dict) -> list[dict]:

    #split the text using these as separators in order of priority
    child_texts  = _split(doc["text"], ["\n\n", "\n", ". ", " "])

    """A long wiki page or PDF manual small chunk of 500 would not be releavnt without the greater context (the parent text)
    -- add the parent text as metdata for the chunk"""

 # coarse parent split — only matters for long docs
    if len(doc["text"]) > config.CHUNK_SIZE * 4:
        parent_texts = _split(doc["text"], ["\n\n"])  # paragraph-level only
    else:
        parent_texts = [doc["text"]]  # whole doc is its own parent

    chunks = []
    for i, t in enumerate(child_texts):

        #check text not empty after cleaning white space
        if not t.strip():
            continue

        parent_idx = min(i * len(parent_texts) // max(len(child_texts), 1), len(parent_texts) - 1)
        c = _make_chunk(doc, t, i, "recursive")
        c["metadata"]["parent_text"] = parent_texts[parent_idx]
        chunks.append(c)
    return chunks


def _split(text: str, seps: list[str]) -> list[str]:
    if len(text) <= config.CHUNK_SIZE:
        return [text]
    sep, rest = seps[0], seps[1:]
    parts, chunks, current = text.split(sep), [], ""
    for part in parts:
        candidate = (current + sep + part).strip() if current else part
        if len(candidate) <= config.CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > config.CHUNK_SIZE and rest:
                chunks.extend(_split(part, rest))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    # add overlap
    if config.CHUNK_OVERLAP and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            overlapped.append((chunks[i-1][-config.CHUNK_OVERLAP:] + " " + chunks[i]).strip())
        return overlapped
    return chunks


def _make_chunk(doc: dict, text: str, idx: int, strategy: str) -> dict:
    return {
        "source":      doc["source"],
        "chunk_index": idx,
        "text":        text,
        "metadata":    {**doc["metadata"], "chunk_strategy": strategy,
                        "chunk_size": len(text)},
    }

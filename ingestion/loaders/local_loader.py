"""ingestion/loaders/local_loader.py — loads .txt and .pdf files from disk."""
import os
from pypdf import PdfReader

#read and extract file contents based on their extension, pattern Registry
# p path of file, read and strip white spaces for txt
# for pdf loop all pages and extract text and join them in a long string, or "" avoids crash transfoming None in empty string
SUPPORTED = {
    ".txt": lambda p: open(p, encoding="utf-8").read().strip(),
    ".pdf": lambda p: "\n\n".join(
        page.extract_text() or "" for page in PdfReader(p).pages
    ).strip(),
}

def load(docs_dir: str) -> list[dict]:
    docs = [] #list

    #for reach file found in folder get extension and check if supported
    for fname in sorted(os.listdir(docs_dir)):
        ext  = os.path.splitext(fname)[1].lower()
        path = os.path.join(docs_dir, fname)
        if ext not in SUPPORTED:
            continue
        try:
            text = SUPPORTED[ext](path) #call function on path of file
            if text:
                docs.append({
                    "source":   fname,
                    "text":     text,
                    "metadata": {"type": "document", "filetype": ext.lstrip("."),
                                 "char_count": len(text)},
                })
                print(f"  ✅ {fname} ({len(text):,} chars)")
        except Exception as e:
            print(f"  ❌ {fname}: {e}")
    return docs

from pathlib import Path


def load_role_documents(directory: str | Path) -> list[dict[str, str]]:
    """Load role knowledge files as plain text documents."""
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"Role knowledge directory not found: {base}")
    docs = []
    for path in sorted(base.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"source": path.name, "text": text})
    if not docs:
        raise ValueError("No non-empty job role documents were found.")
    return docs


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    """Split text into overlapping word chunks to preserve local context."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks

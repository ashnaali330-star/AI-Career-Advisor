from functools import lru_cache


@lru_cache(maxsize=2)
def _model(model_name: str):
    from sentence_transformers import SentenceTransformer
    # Role documents and the embedding model are local assets. Refuse an implicit
    # Hub download here so a missing model fails clearly instead of hanging RAG.
    return SentenceTransformer(model_name, local_files_only=True)


def embed_texts(texts: list[str], model_name: str) -> "object":
    """Create normalized sentence embeddings suitable for cosine search."""
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")
    return _model(model_name).encode(texts, normalize_embeddings=True, convert_to_numpy=True)

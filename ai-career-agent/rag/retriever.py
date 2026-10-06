from rag.document_loader import chunk_text, load_role_documents
from rag.embeddings import embed_texts
from rag.vector_store import RoleVectorStore


class RoleRetriever:
    def __init__(self, docs_dir: str, db_dir: str, embedding_model: str):
        self.docs_dir, self.embedding_model = docs_dir, embedding_model
        self.store = RoleVectorStore(db_dir)

    def ensure_index(self) -> None:
        docs = load_role_documents(self.docs_dir)
        chunks, metadata = [], []
        for doc in docs:
            for i, chunk in enumerate(chunk_text(doc["text"])):
                chunks.append(chunk)
                metadata.append({"source": doc["source"], "text": chunk, "chunk": str(i)})
        vectors = embed_texts(chunks, self.embedding_model)
        self.store.build(vectors, metadata)

    def retrieve(self, query: str, top_k: int = 5, *, build_if_missing: bool = True) -> list[dict[str, str]]:
        if not query or not query.strip():
            raise ValueError("A search query is required for role retrieval.")
        if not self.store.load():
            if not build_if_missing:
                return []
            self.ensure_index()
            if not self.store.load():
                raise RuntimeError("The local role search index could not be loaded after it was built.")
        vector = embed_texts([query], self.embedding_model)
        return self.store.search(vector[0], top_k)

    def retrieve_role(self, role: str, top_k: int = 8) -> list[dict[str, str]]:
        if not role or not role.strip():
            raise ValueError("Select a target job role before searching role requirements.")
        hits = self.retrieve(f"{role} responsibilities required technical skills preferred skills education tools learning topics", top_k=top_k)
        key = role.lower().replace(" ", "_")
        selected = [h for h in hits if key in h["source"].lower() or role.lower() in h["text"].lower()]
        results = selected or hits
        if not results:
            raise RuntimeError(f"No job-role requirements were found for {role}. Check the local data/job_roles documents.")
        return results

from pathlib import Path
import faiss
import numpy as np


class RoleVectorStore:
    """Small persistent FAISS index with parallel chunk metadata."""
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "roles.faiss"
        self.meta_path = self.directory / "roles_metadata.json"
        self.index = None
        self.metadata: list[dict[str, str]] = []

    def build(self, vectors, metadata: list[dict[str, str]]) -> None:
        import json
        array = np.asarray(vectors, dtype="float32")
        if len(array) != len(metadata) or not len(array):
            raise ValueError("Vector and metadata counts must match and be non-empty.")
        self.index = faiss.IndexFlatIP(array.shape[1])
        self.index.add(array)
        self.metadata = metadata
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> bool:
        if self.index is not None:
            return True
        import json
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.metadata):
            self.index = None
            self.metadata = []
            return False
        return True

    def search(self, vector, top_k: int = 5) -> list[dict[str, str]]:
        if self.index is None:
            raise RuntimeError("Vector index has not been built or loaded.")
        scores, ids = self.index.search(np.asarray(vector, dtype="float32").reshape(1, -1), min(top_k, self.index.ntotal))
        return [{**self.metadata[i], "score": float(score)} for score, i in zip(scores[0], ids[0]) if i >= 0]

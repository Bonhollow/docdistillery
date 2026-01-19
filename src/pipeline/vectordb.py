from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import numpy as np

try:
    from qdrant_client import QdrantClient

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


class VectorDB(ABC):
    """
    Abstract interface for Vector Database adapters.
    """

    @abstractmethod
    def create_collection(self, name: str, vector_size: int, distance: str = "Cosine") -> None:
        """Creates a collection/index."""
        pass

    @abstractmethod
    def upsert_chunks(self, collection: str, chunks: List[Dict[str, Any]], embeddings: "np.ndarray") -> None:
        """Helper to upsert matched chunks and embeddings."""
        pass

    @abstractmethod
    def search(self, collection: str, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Searches for nearest neighbors.
        Returns: [{"id": str, "score": float, "payload": dict}]
        """
        pass

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Deletes a collection."""
        pass


class InMemoryAdapter(VectorDB):
    """
    Fallback Vector DB adapter using pure Python and NumPy.
    Suitable for local testing and lightweight tasks.
    """

    def __init__(self):
        self.collections: Dict[str, Dict[str, Any]] = {}

    def create_collection(self, name: str, vector_size: int, distance: str = "Cosine") -> None:
        if distance.lower() != "cosine":
            raise ValueError("InMemoryAdapter currently only supports 'Cosine' distance.")
        self.collections[name] = {
            "vector_size": vector_size,
            "points": [],  # List of {"id": str, "vector": np.ndarray, "payload": dict}
        }

    def upsert_chunks(self, collection: str, chunks: List[Dict[str, Any]], embeddings: "np.ndarray") -> None:
        import numpy as np

        if collection not in self.collections:
            raise ValueError(f"Collection '{collection}' not found.")

        target = self.collections[collection]["points"]
        existing_ids = {p["id"]: i for i, p in enumerate(target)}

        for i, chunk in enumerate(chunks):
            p_id = chunk.get("chunk_id", chunk.get("id"))
            new_point = {
                "id": p_id,
                "vector": np.array(embeddings[i], dtype=np.float32),
                "payload": chunk,
            }
            if p_id in existing_ids:
                target[existing_ids[p_id]] = new_point
            else:
                target.append(new_point)

    def search(self, collection: str, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        import numpy as np

        if collection not in self.collections:
            raise ValueError(f"Collection '{collection}' not found.")

        points = self.collections[collection]["points"]
        if not points:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)

        results = []
        for p in points:
            p_vec = p["vector"]
            p_norm = np.linalg.norm(p_vec)

            # Cosine similarity: (A . B) / (||A|| * ||B||)
            # We assume vectors might not be normalized (though Embedder usually does)
            if q_norm == 0 or p_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(q_vec, p_vec) / (q_norm * p_norm))

            results.append({"id": p["id"], "score": score, "payload": p["payload"]})

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def delete_collection(self, name: str) -> None:
        if name in self.collections:
            del self.collections[name]


try:
    from qdrant_client import QdrantClient

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


class QdrantAdapter(VectorDB):
    """
    Vector DB adapter for Qdrant.
    """

    def __init__(self, location: str = ":memory:"):
        if not HAS_QDRANT:
            raise ImportError("qdrant-client is not installed.")
        self.client = QdrantClient(location=location)

    def create_collection(self, name: str, vector_size: int, distance: str = "Cosine") -> None:
        if not HAS_QDRANT:
            raise ImportError("qdrant-client is not installed.")
        from qdrant_client.models import Distance, VectorParams

        q_dist = Distance.COSINE
        if distance.lower() == "euclidean":
            q_dist = Distance.EUCLID
        elif distance.lower() == "dot":
            q_dist = Distance.DOT

        # Check if collection exists first to mimic recreate behavior
        if self.client.collection_exists(name):
            self.client.delete_collection(name)

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=q_dist),
        )

    def upsert_chunks(self, collection: str, chunks: List[Dict[str, Any]], embeddings: "np.ndarray") -> None:
        if not HAS_QDRANT:
            raise ImportError("qdrant-client is not installed.")
        from qdrant_client.models import PointStruct

        points = []
        for i, chunk in enumerate(chunks):
            p_id = chunk.get("chunk_id", chunk.get("id"))
            points.append(PointStruct(id=p_id, vector=embeddings[i].tolist(), payload=chunk))
        self.client.upsert(collection_name=collection, points=points)

    def search(self, collection: str, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        if not HAS_QDRANT:
            raise ImportError("qdrant-client is not installed.")
        resp = self.client.query_points(collection_name=collection, query=query_vector, limit=limit)
        return [{"id": str(hit.id), "score": hit.score, "payload": hit.payload} for hit in resp.points]

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(collection_name=name)

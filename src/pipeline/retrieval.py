from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pipeline.embed import Embedder
from pipeline.vectordb import VectorDB


class CrossEncoderAdapter(ABC):
    """
    Abstract interface for Cross-Encoder rerankers.
    """

    @abstractmethod
    def rank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks a list of chunks based on the query.
        Each chunk must have 'id' and 'payload'.
        Should update or add 'score'.
        """
        pass


class Retriever:
    """
    Integrates vector search and optional reranking.
    """

    def __init__(
        self, db: VectorDB, collection_name: str, embedder: Embedder, reranker: Optional[CrossEncoderAdapter] = None
    ):
        self.db = db
        self.collection_name = collection_name
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 12, top_k_rerank: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant chunks.

        Args:
            query (str): The search query.
            top_k (int): Number of final results to return.
            top_k_rerank (int): Number of initial results to rerank.

        Returns:
            List[Dict]: List of chunks with 'id', 'score', and 'payload'.
        """
        # 1. Generate query embedding
        query_vector = self.embedder.embed_text(query).tolist()

        # 2. Initial vector search
        # If reranking is applied, we might want to retrieve more than top_k initially
        search_limit = max(top_k, top_k_rerank or 0)
        results = self.db.search(self.collection_name, query_vector, limit=search_limit)

        if not results:
            return []

        # 3. Optional Reranking
        if self.reranker:
            # Rerank either all or top_k_rerank
            to_rerank = results[:top_k_rerank] if top_k_rerank else results
            not_reranked = results[top_k_rerank:] if top_k_rerank else []

            reranked = self.reranker.rank(query, to_rerank)

            # Combine and re-sort by the new scores
            combined = reranked + not_reranked
            combined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            results = combined

        # 4. Return top_k sorted by latest score
        return results[:top_k]

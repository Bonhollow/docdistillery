from typing import Any, Dict, List

import numpy as np
import pytest

from pipeline.retrieval import CrossEncoderAdapter, Retriever
from pipeline.vectordb import InMemoryAdapter


class MockEmbedder:
    """Mock embedder for retrieval tests."""

    def embed_text(self, text: str) -> np.ndarray:
        # Simple deterministic vector
        return np.array([float(len(text)), 0.0], dtype=np.float32)


class MockCrossEncoder(CrossEncoderAdapter):
    """
    Mock reranker that reorders docs based on text length
    (mimicking some arbitrary relevance).
    """

    def rank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for chunk in chunks:
            # Score = length of text in payload
            text = chunk.get("payload", {}).get("text", "")
            chunk["score"] = float(len(text))

        return sorted(chunks, key=lambda x: x["score"], reverse=True)


@pytest.fixture
def setup_retriever():
    db = InMemoryAdapter()
    collection = "test_retrieval"
    db.create_collection(collection, 2)

    chunks = [
        {"chunk_id": "1", "text": "short"},  # len 5
        {"chunk_id": "2", "text": "very long text"},  # len 14
        {"chunk_id": "3", "text": "medium text"},  # len 11
    ]
    embeddings = np.array(
        [
            [10.0, 0.0],
            [11.0, 0.0],
            [12.0, 0.0],
        ]
    )
    db.upsert_chunks(collection, chunks, embeddings)

    embedder = MockEmbedder()
    return db, collection, embedder


def test_retrieval_no_rerank(setup_retriever):
    db, collection, embedder = setup_retriever
    retriever = Retriever(db, collection, embedder)

    # Query "test" (len 4) -> Vector [4, 0]
    # Similarity will favor closest vectors.
    # In our case, higher vector[0] values are closer to [4,0] in normalized space?
    # Wait, InMemoryAdapter uses dot product / norm. [4,0] is same direction as all.
    # Scores will be near 1.0 for all.
    results = retriever.retrieve("test", top_k=2)

    assert len(results) == 2
    assert "id" in results[0]
    assert "score" in results[0]
    assert "payload" in results[0]


def test_retrieval_with_rerank(setup_retriever):
    db, collection, embedder = setup_retriever
    reranker = MockCrossEncoder()
    retriever = Retriever(db, collection, embedder, reranker=reranker)

    results = retriever.retrieve("test", top_k=10)

    # Reranker should sort by text length descending:
    # 2 (len 14), 3 (len 11), 1 (len 5)
    assert results[0]["id"] == "2"
    assert results[1]["id"] == "3"
    assert results[2]["id"] == "1"
    assert results[0]["score"] == 14.0


def test_retrieval_top_k_rerank(setup_retriever):
    db, collection, embedder = setup_retriever
    reranker = MockCrossEncoder()
    retriever = Retriever(db, collection, embedder, reranker=reranker)

    # Vector search returns all (1, 2, 3 have same direction).
    # Let's say top_k_rerank=2. Only the top 2 from vector search get reranked.
    # To be predictable, let's just check if labels are preserved relative to limits.
    results = retriever.retrieve("test", top_k=3, top_k_rerank=2)

    assert len(results) == 3
    # IDs 1, 2, 3 are all equally similar to [4,0] in cosine.
    # InMemoryAdapter returns them in some order (likely [1, 2, 3] from upsert).
    # Reranking top 2 ([1, 2]): "very long text" (2) > "short" (1).
    # Result should be [2, 1, 3]
    assert results[0]["id"] == "2"
    assert results[1]["id"] == "1"
    assert results[2]["id"] == "3"

import pytest

from pipeline.vectordb import HAS_QDRANT, InMemoryAdapter, QdrantAdapter


def test_in_memory_adapter_basic():
    db = InMemoryAdapter()
    collection = "test_col"
    vector_size = 4

    db.create_collection(collection, vector_size)

    import numpy as np

    chunks = [
        {"chunk_id": "1", "meta": "p1"},
        {"chunk_id": "2", "meta": "p2"},
        {"chunk_id": "3", "meta": "p3"},
    ]
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
    ])
    db.upsert_chunks(collection, chunks, embeddings)

    # Search for [1, 0, 0, 0]
    results = db.search(collection, [1.0, 0.0, 0.0, 0.0], limit=2)

    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["id"] == "3"
    assert results[1]["score"] > 0.8
    assert results[0]["payload"]["meta"] == "p1"


def test_in_memory_adapter_upsert_update():
    db = InMemoryAdapter()
    collection = "update_col"
    db.create_collection(collection, 2)

    import numpy as np
    db.upsert_chunks(collection, [{"chunk_id": "v1", "val": 1}], np.array([[1.0, 0.0]]))
    db.upsert_chunks(collection, [{"chunk_id": "v1", "val": 2}], np.array([[0.0, 1.0]]))

    results = db.search(collection, [0.0, 1.0])
    assert results[0]["id"] == "v1"
    assert results[0]["payload"]["val"] == 2


def test_in_memory_adapter_delete():
    db = InMemoryAdapter()
    db.create_collection("to_delete", 2)
    db.delete_collection("to_delete")
    with pytest.raises(ValueError):
        db.search("to_delete", [1, 0])


@pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")
def test_qdrant_adapter_smoke():
    db = QdrantAdapter(location=":memory:")
    collection = "qdrant_smoke"
    db.create_collection(collection, 3)

    import numpy as np
    db.upsert_chunks(
        collection,
        [{"chunk_id": "00000000-0000-0000-0000-000000000001", "tag": "first"}],
        np.array([[1.0, 0.0, 0.0]]),
    )

    results = db.search(collection, [1.0, 0.0, 0.0])
    assert len(results) == 1
    assert results[0]["id"] == "00000000-0000-0000-0000-000000000001"
    assert results[0]["payload"]["tag"] == "first"

    db.delete_collection(collection)

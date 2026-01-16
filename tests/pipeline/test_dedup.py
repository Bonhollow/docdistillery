import numpy as np
import pytest

from pipeline.dedup import cluster_embeddings, select_representatives


def test_cluster_embeddings_kmeans():
    # Use synthetic data to test KMeans fallback
    # 2 clear clusters
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.9, -0.1],  # Cluster 1
            [0.0, 1.0],
            [0.1, 1.1],
            [-0.1, 0.9],  # Cluster 2
        ],
        dtype=np.float32,
    )

    labels = cluster_embeddings(embeddings, method="kmeans", params={"n_clusters": 2})

    assert len(labels) == 6
    # First 3 should have same label, last 3 should have same label
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_select_representatives_centrality():
    chunks = [
        {"id": "c1", "text": "Core concept"},
        {"id": "c2", "text": "Slight variation"},
        {"id": "c3", "text": "Outlier"},
    ]
    # C1 is at [1,0], C2 is at [0.9, 0.1], C3 is further at [0.7, 0.5]
    # Centroid of C1, C2, C3 will be near C1/C2.
    embeddings = np.array(
        [
            [1.0, 0.1],  # c1
            [1.0, 0.0],  # c2 (most central)
            [1.0, -0.1],  # c3
        ],
        dtype=np.float32,
    )
    labels = [0, 0, 0]  # All in same cluster

    reps = select_representatives(chunks, embeddings, labels, top_n_per_cluster=1)

    assert len(reps) == 1
    # C2 is the exact centroid
    assert reps[0]["id"] == "c2"


def test_select_representatives_noise():
    chunks = [{"id": "c1", "text": "Cluster text"}, {"id": "n1", "text": "Noise text"}]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = [0, -1]  # 0 = cluster, -1 = noise

    reps = select_representatives(chunks, embeddings, labels, top_n_per_cluster=1)

    assert len(reps) == 2
    ids = [r["id"] for r in reps]
    assert "c1" in ids
    assert "n1" in ids


def test_select_representatives_validation():
    chunks = [{"id": "1"}]
    embeddings = np.array([[1, 0]])
    labels = [0, 1]  # Wrong length

    with pytest.raises(ValueError, match="Input mismatch"):
        select_representatives(chunks, embeddings, labels)


def test_cluster_embeddings_empty():
    assert cluster_embeddings(np.array([])) == []

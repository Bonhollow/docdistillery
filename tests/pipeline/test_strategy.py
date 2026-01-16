import numpy as np
import pytest

from pipeline.strategy import audit_report, compute_doc_stats, select_strategy


def test_compute_doc_stats_basic():
    chunks = [
        {"text": "Hello world"},  # 2 words -> 2.6 tokens -> 3
        {"text": "This is a test document."},  # 5 words -> 6.5 tokens -> 7
    ]
    # Total tokens expect 10
    stats = compute_doc_stats(chunks)
    assert stats["tokens_estimate"] == 10
    assert stats["n_chunks"] == 2
    assert stats["redundancy_score"] is None


def test_compute_doc_stats_with_embeddings():
    chunks = [{"text": "a"}, {"text": "b"}]
    # Identical embeddings (max redundancy)
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
    stats = compute_doc_stats(chunks, embeddings)
    assert stats["redundancy_score"] == pytest.approx(1.0)


def test_select_strategy():
    # Low redundancy, few chunks -> sequential
    stats = {"redundancy_score": 0.5, "n_chunks": 10}
    assert select_strategy(stats) == "sequential"

    # High redundancy -> clustered
    stats = {"redundancy_score": 0.9, "n_chunks": 10}
    assert select_strategy(stats) == "clustered"

    # Many chunks -> clustered
    stats = {"redundancy_score": 0.2, "n_chunks": 100}
    assert select_strategy(stats) == "clustered"

    # Fallback (no embeddings)
    stats = {"redundancy_score": None, "n_chunks": 20}
    assert select_strategy(stats) == "sequential"
    stats = {"redundancy_score": None, "n_chunks": 60}
    assert select_strategy(stats) == "clustered"


def test_audit_report():
    summary = {"doc_id": "doc1", "provenance": {"part1": ["chunk1"], "part2": ["chunk3"]}}
    clusters = [
        {"cluster_id": 0, "chunk_ids": ["chunk1", "chunk2"]},
        {"cluster_id": 1, "chunk_ids": ["chunk3"]},
        {"cluster_id": 2, "cluster_ids": ["chunk4"]},  # Error in field name in my mental model, fix it
    ]
    # Redefine clusters correctly
    clusters = [
        {"cluster_id": 0, "chunk_ids": ["chunk1", "chunk2"]},
        {"cluster_id": 1, "chunk_ids": ["chunk3"]},
        {"cluster_id": 2, "chunk_ids": ["chunk4"]},
    ]

    report = audit_report(summary, clusters)
    assert report["total_clusters"] == 3
    assert report["covered_clusters"] == 2
    assert report["coverage_ratio"] == pytest.approx(2 / 3)

    details = report["coverage_details"]
    assert details[0]["is_covered"] is True
    assert details[1]["is_covered"] is True
    assert details[2]["is_covered"] is False

import math
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import numpy as np


def compute_doc_stats(chunks: List[Dict[str, Any]], embeddings: Optional["np.ndarray"] = None) -> Dict[str, Any]:
    """
    Computes document statistics for strategy selection and auditing.

    Args:
        chunks: List of document chunks.
        embeddings: Matrix of chunk embeddings.

    Returns:
        Dictionary with tokens_estimate, redundancy_score, n_chunks, and structural_complexity.
    """
    import numpy as np

    total_words = sum(len(c.get("text", "").split()) for c in chunks)
    tokens_estimate = math.ceil(total_words * 1.3)

    redundancy_score = None
    if embeddings is not None and len(embeddings) > 1:
        if len(embeddings) > 100:
            indices = list(range(len(embeddings)))
            pair_similarities = []
            for _ in range(1000):
                idx1, idx2 = random.sample(indices, 2)
                v1, v2 = embeddings[idx1], embeddings[idx2]
                sim = np.dot(v1, v2)
                pair_similarities.append(sim)
            redundancy_score = float(np.mean(pair_similarities))
        else:
            sim_matrix = np.dot(embeddings, embeddings.T)
            indices = np.triu_indices(len(embeddings), k=1)
            redundancy_score = float(np.mean(sim_matrix[indices]))

    structural_complexity = 0
    if chunks:
        avg_chunk_size = sum(len(c.get("text", "")) for c in chunks) / len(chunks)
        if avg_chunk_size < 500:
            structural_complexity = 1
        elif avg_chunk_size > 3000:
            structural_complexity = 3
        else:
            structural_complexity = 2

    return {
        "tokens_estimate": tokens_estimate,
        "redundancy_score": redundancy_score,
        "n_chunks": len(chunks),
        "structural_complexity": structural_complexity,
    }


def select_strategy(stats: Dict[str, Any], config: Optional[Dict[str, Any]] = None, query: Optional[str] = None) -> str:
    """
    Selects the optimal summarization strategy based on document statistics.

    Args:
        stats: Document statistics from compute_doc_stats.
        config: Configuration override with thresholds.
        query: Optional user query for insight-driven mode.

    Returns:
        Strategy name: "sequential" | "clustered" | "insight_driven"
    """
    if config is None:
        config = {}

    redundancy_thresh = config.get("redundancy_thresh", 0.75)
    n_chunks_thresh = config.get("n_chunks_thresh", 30)
    complexity_thresh = config.get("complexity_thresh", 2)

    redundancy = stats.get("redundancy_score")
    n_chunks = stats.get("n_chunks", 0)
    complexity = stats.get("structural_complexity", 2)

    if query and len(query.strip()) > 5:
        return "insight_driven"

    if redundancy is None:
        if n_chunks >= n_chunks_thresh:
            return "clustered"
        return "sequential"

    if redundancy >= redundancy_thresh:
        return "clustered"

    if n_chunks >= n_chunks_thresh and complexity >= complexity_thresh:
        return "clustered"

    if complexity >= 3 and n_chunks >= 15:
        return "clustered"

    return "sequential"


def recommend_chunking(strategy: str, doc_length: int, stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recommend chunking parameters based on strategy and document.

    Args:
        strategy: Selected summarization strategy.
        doc_length: Document length in characters.
        stats: Document statistics.

    Returns:
        Dictionary with recommended chunk_size_chars, overlap_chars, and deduplicate.
    """
    if strategy == "insight_driven":
        return {"chunk_size_chars": 1500, "overlap_chars": 200, "deduplicate": True}

    if strategy == "clustered":
        return {"chunk_size_chars": 2000, "overlap_chars": 0, "deduplicate": True}

    return {"chunk_size_chars": 2500, "overlap_chars": 0, "deduplicate": False}


def audit_report(summary_obj: Dict[str, Any], clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates a coverage report comparing the summary provenance with document clusters.

    Args:
        summary_obj (Dict): The summary object with 'provenance'.
        clusters (List[Dict]): List of clusters [{"cluster_id": int, "chunk_ids": List[str]}].

    Returns:
        Dict: Machine-readable audit report.
    """
    provenance = summary_obj.get("provenance", {})
    # Flatten all chunk IDs present in the summary
    chunks_in_summary = set()
    for chunk_ids in provenance.values():
        if isinstance(chunk_ids, list):
            for cid in chunk_ids:
                chunks_in_summary.add(cid)
        else:
            chunks_in_summary.add(chunk_ids)

    report = {
        "summary_id": summary_obj.get("doc_id", "unknown"),
        "total_clusters": len(clusters),
        "covered_clusters": 0,
        "coverage_details": [],
    }

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        cluster_chunk_ids = cluster["chunk_ids"]

        # A cluster is "covered" if any of its member chunks are referenced in the summary provenance
        covered = any(cid in chunks_in_summary for cid in cluster_chunk_ids)

        if covered:
            report["covered_clusters"] += 1

        report["coverage_details"].append(
            {
                "cluster_id": cluster_id,
                "is_covered": covered,
                "n_chunks": len(cluster_chunk_ids),
                "representative_present": list(set(cluster_chunk_ids) & chunks_in_summary),
            }
        )

    report["coverage_ratio"] = report["covered_clusters"] / len(clusters) if clusters else 1.0
    return report

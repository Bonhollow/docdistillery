from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import numpy as np


def cluster_embeddings(embeddings: "np.ndarray", method: str = "hdbscan", params: Optional[dict] = None) -> List[int]:
    """
    Groups embeddings into clusters.

    Args:
        embeddings: The embeddings to cluster (N, D).
        method: Clustering method ('hdbscan' or 'kmeans').
        params: Optional hyperparameters for the algorithm.

    Returns:
        List[int]: Cluster labels for each embedding.
    """

    if len(embeddings) == 0:
        return []

    params = params or {}

    if method == "hdbscan":
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(**(params or {"min_cluster_size": 2}))
            return clusterer.fit_predict(embeddings).tolist()
        except ImportError:
            method = "kmeans"

    from sklearn.cluster import KMeans

    n_clusters = (params or {}).get("n_clusters", max(2, len(embeddings) // 5))
    clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = clusterer.fit_predict(embeddings)
    return labels.tolist()


def select_representatives(
    chunks: List[Dict[str, Any]], embeddings: "np.ndarray", labels: List[int], top_n_per_cluster: int = 1
) -> List[Dict[str, Any]]:
    """
    Selects representative chunks from each cluster based on centrality.

    Args:
        chunks: The chunk dictionaries.
        embeddings: The embeddings corresponding to chunks (N, D).
        labels: Cluster labels for each chunk.
        top_n_per_cluster: How many chunks to select per cluster.

    Returns:
        List[Dict]: The list of selected representative chunks.
    """
    import numpy as np

    if not chunks:
        return []

    if len(chunks) != len(embeddings) or len(chunks) != len(labels):
        msg = f"chunks({len(chunks)}), embeddings({len(embeddings)}), labels({len(labels)})"
        raise ValueError(f"Input mismatch: {msg} must have same length.")

    representatives = []
    unique_labels = sorted(list(set(labels)))

    for label in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]

        if label == -1:
            for idx in indices:
                representatives.append(chunks[idx])
            continue

        cluster_embeddings = embeddings[indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        norm_centroid = np.linalg.norm(centroid)
        if norm_centroid > 0:
            centroid = centroid / norm_centroid

        distances = []
        for i, idx in enumerate(indices):
            emb = cluster_embeddings[i]
            norm_emb = np.linalg.norm(emb)
            if norm_emb > 0 and norm_centroid > 0:
                similarity = np.dot(emb, centroid) / (norm_emb * norm_centroid)
            else:
                similarity = 0.0
            distances.append((1.0 - similarity, idx))

        distances.sort(key=lambda x: x[0])

        for _, idx in distances[:top_n_per_cluster]:
            representatives.append(chunks[idx])

    return representatives


def deduplicate_by_similarity(
    chunks: List[Dict[str, Any]], embeddings: "np.ndarray", threshold: float = 0.85, keep_longest: bool = True
) -> List[Dict[str, Any]]:
    """
    Remove chunks that are highly similar to others.

    Args:
        chunks: List of chunk dictionaries.
        embeddings: Embeddings for each chunk.
        threshold: Similarity threshold above which chunks are considered duplicates.
        keep_longest: Whether to keep the longest chunk from each duplicate group.

    Returns:
        List[Dict]: Deduplicated list of chunks.
    """
    import numpy as np

    if len(chunks) < 2 or len(embeddings) < 2:
        return chunks

    n = len(chunks)
    similarity_matrix = np.dot(embeddings, embeddings.T)

    to_remove = set()
    processed = set()

    for i in range(n):
        if i in processed or i in to_remove:
            continue

        group = [i]
        processed.add(i)

        for j in range(i + 1, n):
            if j in processed or j in to_remove:
                continue
            if similarity_matrix[i, j] >= threshold:
                group.append(j)
                processed.add(j)

        if len(group) > 1:
            if keep_longest:
                longest_idx = max(group, key=lambda x: len(chunks[x].get("text", "")))
            else:
                longest_idx = group[0]

            for idx in group:
                if idx != longest_idx:
                    to_remove.add(idx)

    return [c for idx, c in enumerate(chunks) if idx not in to_remove]


def remove_near_duplicates(
    chunks: List[Dict[str, Any]], threshold_ngrams: int = 5, jaccard_threshold: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Remove chunks that have significant n-gram overlap.

    Args:
        chunks: List of chunk dictionaries.
        threshold_ngrams: N-gram size for comparison.
        jaccard_threshold: Jaccard similarity threshold for duplicate detection.

    Returns:
        List[Dict]: Deduplicated list of chunks.
    """
    from collections import Counter

    if len(chunks) < 2:
        return chunks

    def get_ngrams(text: str, n: int) -> Counter:
        words = text.lower().split()
        return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))

    all_ngrams = [get_ngrams(c.get("text", ""), threshold_ngrams) for c in chunks]

    def jaccard_similarity(set1: Counter, set2: Counter) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = sum((set1 & set2).values())
        union = sum((set1 | set2).values())
        return intersection / union if union > 0 else 0.0

    to_remove = set()
    n = len(chunks)

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            similarity = jaccard_similarity(all_ngrams[i], all_ngrams[j])
            if similarity >= jaccard_threshold:
                longer_i = len(chunks[i].get("text", "")) > len(chunks[j].get("text", ""))
                to_remove.add(j if longer_i else i)
                break

    return [c for idx, c in enumerate(chunks) if idx not in to_remove]

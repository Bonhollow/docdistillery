import re
from typing import Any, Dict, List, Optional


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs while preserving structure.

    Args:
        text: The input text to split.

    Returns:
        List of paragraph strings.
    """
    if not text:
        return []

    paragraphs = re.split(r"\n{2,}", text)
    result = []
    for para in paragraphs:
        para = para.strip()
        if para:
            result.append(para)
    return result


def _merge_small_paragraphs(paragraphs: List[str], min_length: int = 100) -> List[str]:
    """
    Merge very small paragraphs with the following paragraph.

    Args:
        paragraphs: List of paragraph strings.
        min_length: Minimum length for a paragraph to remain separate.

    Returns:
        List of merged paragraphs.
    """
    if not paragraphs:
        return []

    merged = []
    buffer = ""

    for para in paragraphs:
        if len(para) < min_length and buffer:
            buffer = buffer + " " + para
        else:
            if buffer:
                merged.append(buffer)
            buffer = para

    if buffer:
        merged.append(buffer)

    return merged


def chunk_text(
    text: str, chunk_size_chars: int = 2000, overlap_chars: int = 0, merge_small_paragraphs: bool = False
) -> List[Dict[str, Any]]:
    """
    Breaks text into chunks with improved semantic boundaries.
    Uses paragraph-based splitting without overlap to avoid duplicate processing.

    Args:
        text: The input text to chunk.
        chunk_size_chars: Targeted maximum size of each chunk.
        overlap_chars: Overlap between consecutive chunks (default 0 to avoid duplicates).
        merge_small_paragraphs: Whether to merge very small paragraphs.

    Returns:
        List of dicts with 'text', 'start_char', and 'end_char'.
    """
    if not text:
        return []

    paragraphs = split_into_paragraphs(text)

    if merge_small_paragraphs:
        paragraphs = _merge_small_paragraphs(paragraphs, min_length=100)

    if not paragraphs:
        return []

    chunks = []
    current_chunk = ""
    current_start = 0
    text_pos = 0

    for para in paragraphs:
        para_len = len(para)
        para_start = text.find(para, text_pos)
        if para_start == -1:
            para_start = text_pos

        if len(current_chunk) + para_len <= chunk_size_chars:
            if current_chunk:
                current_chunk += " " + para
            else:
                current_chunk = para
                current_start = para_start
        else:
            if current_chunk:
                chunks.append(
                    {"text": current_chunk, "start_char": current_start, "end_char": current_start + len(current_chunk)}
                )

            current_chunk = para
            current_start = para_start

        text_pos = para_start + para_len

    if current_chunk:
        chunks.append(
            {"text": current_chunk, "start_char": current_start, "end_char": current_start + len(current_chunk)}
        )

    return chunks


def deduplicate_chunks(
    chunks: List[Dict[str, Any]], similarity_threshold: float = 0.85, embedder: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Remove chunks that are highly similar to others.
    Uses embeddings to compute semantic similarity.

    Args:
        chunks: List of chunk dictionaries.
        similarity_threshold: Threshold above which chunks are considered duplicates.
        embedder: Optional Embedder instance for computing embeddings.

    Returns:
        Deduplicated list of chunks.
    """
    if len(chunks) < 2:
        return chunks

    if embedder is None:
        return chunks

    texts = [c.get("text", "") for c in chunks]

    try:
        from pipeline.embed import Embedder

        if embedder is True or embedder is None:
            embedder = Embedder()

        embeddings = embedder.embed_text(texts)

        import numpy as np

        similarity_matrix = np.dot(embeddings, embeddings.T)

        n = len(chunks)
        to_remove = set()

        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                if similarity_matrix[i, j] >= similarity_threshold:
                    longer_i = len(texts[i]) > len(texts[j])
                    if longer_i:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break

        result = [c for idx, c in enumerate(chunks) if idx not in to_remove]
        return result

    except Exception:
        return chunks


def chunk_pages(
    pages: List[Dict[str, Any]],
    doc_id: str,
    chunk_size_chars: int = 2000,
    overlap_chars: int = 0,
    deduplicate: bool = False,
    embedder: Optional[Any] = None,
    pages_per_chunk: int = 0,
) -> List[Dict[str, Any]]:
    """
    Chunks whole document pages and assigns deterministic IDs.
    Preserves document order (page order, then chunk index).

    Args:
        pages: List of pages from ingest.py.
        doc_id: The document ID.
        chunk_size_chars: Targeted maximum size of each chunk.
        overlap_chars: Overlap between consecutive chunks.
        deduplicate: Whether to remove semantically similar chunks.
        embedder: Optional Embedder instance for deduplication.
        pages_per_chunk: Number of pages to combine per chunk (0 = use chunk_size_chars).

    Returns:
        List of chunks with deterministic IDs and page metadata.
    """
    all_chunks = []
    sorted_pages = sorted(pages, key=lambda x: x.get("page", 0))

    if pages_per_chunk > 0:
        combined_chunks = []
        current_chunk = ""
        current_start_page = 1
        page_nums = []
        char_count = 0

        for page_dict in sorted_pages:
            page_num = page_dict.get("page", 0)
            text = page_dict.get("text", "")
            text_len = len(text)

            if text_len == 0:
                continue

            if char_count + text_len <= chunk_size_chars:
                if current_chunk:
                    current_chunk += "\n\n" + text
                else:
                    current_chunk = text
                    current_start_page = page_num
                page_nums.append(page_num)
                char_count += text_len
            else:
                if current_chunk:
                    combined_chunks.append(
                        {
                            "text": current_chunk,
                            "start_page": current_start_page,
                            "end_page": page_nums[-1] if page_nums else page_num,
                            "pages": page_nums.copy(),
                        }
                    )

                current_chunk = text
                current_start_page = page_num
                page_nums = [page_num]
                char_count = text_len

        if current_chunk:
            combined_chunks.append(
                {
                    "text": current_chunk,
                    "start_page": current_start_page,
                    "end_page": page_nums[-1] if page_nums else sorted_pages[-1].get("page", 0) if sorted_pages else 0,
                    "pages": page_nums.copy(),
                }
            )

        for i, combined in enumerate(combined_chunks):
            chunk = {
                "text": combined["text"],
                "chunk_id": f"{doc_id}_c{i}",
                "start_page": combined["start_page"],
                "end_page": combined["end_page"],
                "pages": combined["pages"],
            }
            all_chunks.append(chunk)
    else:
        for page_dict in sorted_pages:
            page_num = page_dict.get("page", 0)
            text = page_dict.get("text", "")

            page_chunks = chunk_text(text, chunk_size_chars, overlap_chars)

            for i, chunk in enumerate(page_chunks):
                chunk["chunk_id"] = f"{doc_id}_p{page_num}_c{i}"
                chunk["page"] = page_num
                all_chunks.append(chunk)

    if deduplicate and len(all_chunks) > 2:
        all_chunks = deduplicate_chunks(all_chunks, embedder=embedder)

    return all_chunks

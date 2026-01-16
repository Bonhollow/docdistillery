import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


MICRO_SUMMARY_PROMPT = """You are creating a micro-summary for a larger document synthesis.
Focus on unique information in this segment that won't be covered in other segments.
- Include specific facts, examples, and key insights
- Omit generic statements that could apply to any document
- Output: 2-4 sentences maximum
- Be concise but informative

Text segment:
{text}
"""


INTEGRATION_PROMPT = """You are synthesizing multiple partial summaries into a coherent narrative.
Merge overlapping information and remove duplicates while preserving all unique insights.

Partial summaries:
{summaries}

Requirements:
1. Merge sentences that convey the same information
2. Remove duplicate facts and repeated phrases
3. Maintain logical flow between ideas
4. Preserve all unique insights from every summary
5. Output as a single cohesive narrative
6. Use professional, academic tone

Synthesized summary:"""


DEDUP_PROMPT = """Remove duplicate information from the following text while preserving all unique insights.
Combine repeated concepts into single statements.

Text:
{text}

Output: Clean text without repetition, maintaining all unique information."""


SECTION_TITLE_PROMPT = """Generate a brief, descriptive title (3-5 words) for this document section based on its content.
The title should be informative and specific.

Content summary:
{summary}

Title:"""


class SummarizerAdapter(ABC):
    """
    Abstract interface for chunk-level summarizers.
    """

    @abstractmethod
    def summarize_chunk(self, text: str, mode: str = "abstractive") -> str:
        """
        Summarizes a single text chunk.
        """
        pass


class BasicSummarizer(SummarizerAdapter):
    """
    Simple implementation that returns the first few sentences.
    """

    def summarize_chunk(self, text: str, mode: str = "abstractive") -> str:
        return text[:200] + "..." if len(text) > 200 else text


class LLMSummarizer(SummarizerAdapter):
    """
    Abstractive summarizer using an LLM.
    """

    def __init__(self, llm: "LLMAdapter"):
        self.llm = llm

    def summarize_chunk(self, text: str, mode: str = "abstractive", max_output_tokens: int = 1000) -> str:
        if mode == "extractive":
            prompt = f"Identify the most important sentences in this text and return them verbatim:\n\n{text}"
        else:
            prompt = MICRO_SUMMARY_PROMPT.format(text=text)

        return self.llm.generate(prompt, max_tokens=max_output_tokens)


class LLMAdapter(ABC):
    """
    Abstract interface for LLM-based polishing.
    """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        """
        Generates text from a prompt.
        """
        pass


def count_tokens_approx(text: str) -> int:
    """
    Approximates token count using a word-based heuristic (1 word ≈ 1.3 tokens).
    """
    words = text.split()
    return math.ceil(len(words) * 1.3)


def _generate_section_title(content: str, llm: Optional[LLMAdapter] = None) -> str:
    """
    Generate a descriptive title for a section based on its content.

    Args:
        content: The section content.
        llm: Optional LLM for title generation.

    Returns:
        A brief, descriptive title.
    """
    if llm is None:
        return "Summary"

    try:
        prompt = SECTION_TITLE_PROMPT.format(summary=content[:500])
        title = llm.generate(prompt, max_tokens=20)
        title = title.strip().strip('"').strip("'")
        if len(title) > 50:
            title = title[:50] + "..."
        return title if title else "Summary"
    except Exception:
        return "Summary"


def group_chunks_by_similarity(
    chunks: List[Dict[str, Any]],
    embeddings: Optional["np.ndarray"] = None,
    max_groups: int = 10,
    similarity_threshold: float = 0.85,
) -> List[List[Dict[str, Any]]]:
    """
    Group semantically related chunks together using embeddings.
    Creates a fixed number of groups for consistent output.

    Args:
        chunks: List of chunk dictionaries.
        embeddings: Pre-computed embeddings for chunks.
        max_groups: Maximum number of groups to create.
        similarity_threshold: Minimum similarity to group chunks together.

    Returns:
        List of groups, where each group is a list of related chunks.
    """
    if len(chunks) <= 1:
        return [chunks] if chunks else []

    if embeddings is None:
        n = len(chunks)
        group_size = max(1, n // max_groups)
        return [chunks[i : i + group_size] for i in range(0, n, group_size)]

    try:
        import numpy as np
        from sklearn.cluster import KMeans

        n = len(chunks)
        n_clusters = min(max_groups, n)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)

        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        groups = []
        for label in sorted(clusters.keys()):
            indices = clusters[label]
            group = [chunks[idx] for idx in indices]
            groups.append(group)

        return groups

    except Exception:
        n = len(chunks)
        group_size = max(1, n // max_groups)
        return [chunks[i : i + group_size] for i in range(0, n, group_size)]

    try:
        import numpy as np

        n = len(chunks)
        similarity_matrix = np.dot(embeddings, embeddings.T)

        visited = [False] * n
        groups = []

        for i in range(n):
            if visited[i]:
                continue

            group = [i]
            visited[i] = True

            similarities = similarity_matrix[i]

            for j in range(i + 1, n):
                if not visited[j] and similarities[j] > 0.75:
                    if len(group) < max_group_size:
                        group.append(j)
                        visited[j] = True

            groups.append([chunks[idx] for idx in group])

        for i in range(n):
            if not visited[i]:
                groups.append([chunks[i]])
                visited[i] = True

        return groups

    except Exception:
        return [chunks]


def synthesize(
    chunks: List[Dict[str, Any]],
    summarizer: Optional[SummarizerAdapter] = None,
    mode: str = "abstractive",
    llm: Optional[LLMAdapter] = None,
    target_tokens: int = 8000,
    detail_level: str = "standard",
    embeddings: Optional["np.ndarray"] = None,
) -> Dict[str, Any]:
    """
    Summarizes document chunks into a structured synthesis.
    Produces comprehensive output while avoiding excessive repetition.

    Args:
        chunks: Chunks with 'chunk_id' and 'text'.
        summarizer: Adapter for micro-summaries.
        mode: 'abstractive' or 'extractive'.
        llm: Optional LLM for final polishing.
        target_tokens: Soft upper bound for the final summary length.
        detail_level: 'brief', 'standard', or 'detailed'.
        embeddings: Optional pre-computed embeddings for grouping.

    Returns:
        Structured summary with TL;DR, sections, and provenance.
    """
    if not chunks:
        return {"tldr": "", "executive": [], "sections": [], "provenance": {}}

    if summarizer is None:
        summarizer = BasicSummarizer()

    max_chunk_tokens = 500 if detail_level == "detailed" else 300

    n_groups = 5 if detail_level == "brief" else (8 if detail_level == "standard" else 12)
    groups = group_chunks_by_similarity(chunks, embeddings, max_groups=n_groups)

    group_summaries = []
    provenance = {}

    for group_idx, group in enumerate(groups):
        group_texts = [c.get("text", "") for c in group]
        combined_text = " ".join(group_texts)

        if isinstance(summarizer, LLMSummarizer):
            micro_summary = summarizer.summarize_chunk(combined_text, mode=mode, max_output_tokens=max_chunk_tokens)
        else:
            micro_summary = summarizer.summarize_chunk(combined_text, mode=mode)

        chunk_ids = [c.get("chunk_id", f"unknown_{i}") for i, c in enumerate(group)]
        provenance[f"section_{group_idx + 1}"] = chunk_ids
        group_summaries.append(micro_summary)

    if llm and len(group_summaries) > 1:
        try:
            combined_summaries = "\n\n".join(group_summaries)

            final_sections = []
            for idx, summary in enumerate(group_summaries):
                title = f"Section {idx + 1}"
                final_sections.append({"title": title, "content": summary})

            combined_all = " ".join(group_summaries)
            tldr = llm.generate(
                "Create a very brief TL;DR (1-2 sentences, max 30 words) capturing the main topic: "
                + combined_all[:2000],
                max_tokens=50,
            )

            exec_summary = llm.generate(
                "Create a concise executive summary (3-5 bullet points) covering all key themes. Use professional tone:\n\n"
                + combined_all[:3000],
                max_tokens=300,
            )
            executive = [s.strip() for s in exec_summary.split("\n") if s.strip() and len(s.strip()) > 10]

        except Exception:
            final_sections = [{"title": f"Section {i + 1}", "content": s} for i, s in enumerate(group_summaries)]
            tldr = group_summaries[0][:100] if group_summaries else ""
            executive = group_summaries[:5]
    else:
        final_sections = [{"title": f"Section {i + 1}", "content": s} for i, s in enumerate(group_summaries)]
        tldr = group_summaries[0][:100] if group_summaries else ""
        executive = group_summaries[:5]

    return {"tldr": tldr, "executive": executive, "sections": final_sections, "provenance": provenance}

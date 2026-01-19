"""
Document summarization with ordered processing and proportional output.

Key Design Decisions:
- Process chunks SEQUENTIALLY to preserve document order
- Use sliding window (3-5 chunks) for better context
- Target 20-30% output length
- Apply deduplication AFTER summarization
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SummarizerAdapter(ABC):
    """Abstract interface for chunk-level summarizers."""

    @abstractmethod
    def summarize_chunk(self, text: str, target_length: int = 500) -> str:
        """Summarize a single text chunk."""
        pass


class BasicSummarizer(SummarizerAdapter):
    """Extractive fallback summarizer (no LLM required)."""

    def summarize_chunk(self, text: str, target_length: int = 500) -> str:
        """Extract key sentences from text."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score sentences by length and position
        scored = []
        for i, sent in enumerate(sentences):
            # Prefer longer sentences and earlier ones
            score = len(sent.split()) * (1 - i / (len(sentences) + 1) * 0.3)
            scored.append((score, sent))
        
        # Sort by score and take top sentences
        scored.sort(reverse=True)
        
        result = []
        current_length = 0
        for _, sent in scored:
            if current_length + len(sent) > target_length * 4:  # chars approximation
                break
            result.append(sent)
            current_length += len(sent)
        
        # Return in original order
        ordered = [s for s in sentences if s in result]
        return " ".join(ordered) if ordered else text[:target_length * 4]


class LLMSummarizer(SummarizerAdapter):
    """LLM-based abstractive summarizer."""

    def __init__(self, llm: "LLMAdapter"):
        self.llm = llm

    def summarize_chunk(self, text: str, target_length: int = 500) -> str:
        """Generate abstractive summary of text."""
        if not text or len(text.strip()) < 50:
            return text
        
        try:
            result = self.llm.generate(text, max_tokens=target_length)
            if result and len(result.strip()) > 20:
                return result.strip()
        except Exception:
            pass
        
        # Fallback to extractive
        return BasicSummarizer().summarize_chunk(text, target_length)


class LLMAdapter(ABC):
    """Abstract interface for LLM-based text generation."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        pass


def _deduplicate_text(text: str) -> str:
    """Remove duplicate sentences from text."""
    if not text or len(text) < 100:
        return text
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = []
    
    for sent in sentences:
        # Normalize for comparison
        normalized = sent.lower().strip()
        if len(normalized) < 20:
            unique.append(sent)
            continue
        
        # Check for duplicates
        is_dup = False
        for seen_sent in seen:
            # Check substring containment
            if normalized in seen_sent or seen_sent in normalized:
                is_dup = True
                break
        
        if not is_dup:
            unique.append(sent)
            seen.add(normalized)
    
    return " ".join(unique)


def _ensure_complete_sentences(text: str) -> str:
    """Ensure text ends with complete sentences."""
    if not text:
        return text
    
    # Find the last sentence boundary
    last_period = text.rfind('. ')
    last_question = text.rfind('? ')
    last_exclaim = text.rfind('! ')
    
    cut_pos = max(last_period, last_question, last_exclaim)
    
    if cut_pos > len(text) * 0.7:
        return text[:cut_pos + 1].strip()
    
    # If text already ends with punctuation, keep it
    if text.rstrip()[-1] in '.!?':
        return text.rstrip()
    
    return text


def synthesize(
    chunks: List[Dict[str, Any]],
    summarizer: Optional[SummarizerAdapter] = None,
    llm: Optional["LLMAdapter"] = None,
    detail_level: str = "standard",
    mode: str = "abstractive",
    target_tokens: int = 8000,
    embeddings: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Synthesize document chunks into a structured summary.
    
    Key Features:
    - Processes chunks in document order (no clustering)
    - Uses sliding window for context
    - Targets 20-30% output length
    - Deduplicates after generation
    
    Args:
        chunks: List of chunks with 'text' and 'chunk_id'
        summarizer: Adapter for chunk summarization
        llm: LLM adapter for polish/TLDR
        detail_level: "brief", "standard", or "detailed"
        mode: "abstractive" or "extractive"
        target_tokens: Soft limit for output tokens
        embeddings: Ignored (kept for compatibility)
    
    Returns:
        Dict with 'tldr', 'executive', 'sections', 'provenance'
    """
    if not chunks:
        return {"tldr": "", "executive": [], "sections": [], "provenance": {}}
    
    if summarizer is None:
        summarizer = BasicSummarizer()
    
    # Configuration based on detail level
    config = {
        "brief": {"window_size": 5, "target_ratio": 0.15, "max_sections": 8},
        "standard": {"window_size": 3, "target_ratio": 0.25, "max_sections": 15},
        "detailed": {"window_size": 2, "target_ratio": 0.35, "max_sections": 30},
    }
    cfg = config.get(detail_level, config["standard"])
    
    # Calculate input size
    total_input_chars = sum(len(c.get("text", "")) for c in chunks)
    target_output_chars = int(total_input_chars * cfg["target_ratio"])
    chars_per_section = max(500, target_output_chars // cfg["max_sections"])
    
    # Process chunks in order using sliding window
    sections = []
    provenance = {}
    window_size = cfg["window_size"]
    
    i = 0
    while i < len(chunks):
        # Get window of chunks
        window_end = min(i + window_size, len(chunks))
        window_chunks = chunks[i:window_end]
        
        # Combine text from window
        combined_text = " ".join(c.get("text", "") for c in window_chunks)
        
        if not combined_text.strip():
            i += window_size
            continue
        
        # Calculate target length for this section
        target_length = min(chars_per_section // 4, 800)  # Convert to words approx
        
        # Generate summary for this window
        try:
            if isinstance(summarizer, LLMSummarizer):
                summary = summarizer.summarize_chunk(combined_text, target_length=target_length)
            else:
                summary = summarizer.summarize_chunk(combined_text, target_length=target_length)
        except Exception:
            summary = combined_text[:chars_per_section]
        
        # Ensure complete sentences
        summary = _ensure_complete_sentences(summary)
        
        if summary and len(summary.strip()) > 10:
            section_num = len(sections) + 1
            sections.append({
                "title": f"Section {section_num}",
                "content": summary
            })
            
            # Track provenance
            chunk_ids = [c.get("chunk_id", f"chunk_{j}") for j, c in enumerate(window_chunks)]
            provenance[f"section_{section_num}"] = chunk_ids
        
        i += window_size
    
    if not sections:
        return {"tldr": "", "executive": [], "sections": [], "provenance": {}}
    
    # Deduplicate sections
    for section in sections:
        section["content"] = _deduplicate_text(section["content"])
    
    # Generate executive summary (sample from across document)
    n_exec = min(5, len(sections))
    step = max(1, len(sections) // n_exec)
    exec_indices = [i * step for i in range(n_exec)]
    executive = [sections[i]["content"][:500] for i in exec_indices if i < len(sections)]
    
    # Generate TLDR
    tldr = ""
    if llm and sections:
        try:
            # Combine all section summaries for TLDR
            combined = " ".join(s["content"][:300] for s in sections[:10])
            tldr = llm.generate(
                f"Create a 2-3 sentence summary of this document: {combined[:3000]}",
                max_tokens=100
            )
        except Exception:
            pass
    
    if not tldr:
        # Fallback TLDR
        tldr = sections[0]["content"][:200] if sections else ""
        if not tldr.rstrip().endswith(('.', '!', '?')):
            tldr = _ensure_complete_sentences(tldr)
    
    return {
        "tldr": tldr,
        "executive": executive,
        "sections": sections,
        "provenance": provenance
    }

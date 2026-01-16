from pipeline.summarize import LLMAdapter, SummarizerAdapter, synthesize


class MockSummarizer(SummarizerAdapter):
    """Deterministic mock summarizer for testing."""

    def summarize_chunk(self, text: str, mode: str = "abstractive") -> str:
        return f"Summary: {text[:30]}"


class MockLLM(LLMAdapter):
    """Deterministic mock LLM for testing."""

    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        if "TL;DR sentence" in prompt or "one-sentence TL;DR" in prompt:
            return "Polished TL;DR"
        if "executive summary" in prompt.lower():
            return "Polished Point 1\nPolished Point 2"
        if "Synthesized summary" in prompt or "merge" in prompt.lower():
            return "Integrated summary content"
        return "Polished Output"


def test_synthesize_basic():
    chunks = [
        {"chunk_id": "doc1_c1", "text": "This is the first chunk of text that needs summarizing."},
        {"chunk_id": "doc1_c2", "text": "This is the second chunk of text with more details."},
    ]
    summarizer = MockSummarizer()

    result = synthesize(chunks, summarizer=summarizer)

    assert result["tldr"].startswith("Summary:") or result["tldr"].startswith("Polished") or result["tldr"] != ""
    assert len(result["sections"]) >= 1
    assert "provenance" in result


def test_synthesize_with_llm():
    chunks = [
        {"chunk_id": "c1", "text": "Lead chunk text here."},
    ]
    summarizer = MockSummarizer()
    llm = MockLLM()

    result = synthesize(chunks, summarizer=summarizer, llm=llm)

    assert result["tldr"] != "" or "Polished" in result["tldr"]
    assert len(result["sections"]) >= 1


def test_synthesize_token_budget():
    chunks = [
        {"chunk_id": "c1", "text": "Long text " * 100},
        {"chunk_id": "c2", "text": "Extra text"},
    ]
    summarizer = MockSummarizer()

    result = synthesize(chunks, summarizer=summarizer, target_tokens=2)

    assert len(result["sections"]) >= 1


def test_synthesize_empty():
    result = synthesize([], summarizer=MockSummarizer())
    assert result["tldr"] == ""
    assert result["sections"] == []
    assert result["provenance"] == {}


def test_synthesize_groups_chunks():
    chunks = [
        {"chunk_id": "c1", "text": "First topic discussion."},
        {"chunk_id": "c2", "text": "More about the first topic."},
        {"chunk_id": "c3", "text": "A completely different topic."},
    ]
    summarizer = MockSummarizer()

    result = synthesize(chunks, summarizer=summarizer)

    assert len(result["sections"]) >= 1
    combined = " ".join(s["content"] for s in result["sections"])
    assert "First topic" in combined or "Summary:" in combined

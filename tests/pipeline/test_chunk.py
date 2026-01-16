from pipeline.chunk import chunk_pages, chunk_text


def test_chunk_text_basic():
    text = "Sentence one. Sentence two. Sentence three."
    chunks = chunk_text(text, chunk_size_chars=15, overlap_chars=5)

    assert len(chunks) >= 1
    assert "start_char" in chunks[0]
    assert "end_char" in chunks[0]


def test_chunk_text_overlap():
    text = "01234567890123456789"
    chunks = chunk_text(text, chunk_size_chars=10, overlap_chars=5)

    assert len(chunks) >= 1


def test_chunk_text_newlines():
    text = "First line\n\nSecond line\n\nThird line"
    chunks = chunk_text(text, chunk_size_chars=15, overlap_chars=5)

    assert len(chunks) >= 1
    combined = " ".join(c["text"] for c in chunks)
    assert "First line" in combined
    assert "Second line" in combined
    assert "Third line" in combined


def test_chunk_text_paragraphs():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, chunk_size_chars=100, overlap_chars=0)

    assert len(chunks) >= 1
    combined = " ".join(c["text"] for c in chunks)
    assert "First paragraph" in combined
    assert "Second paragraph" in combined
    assert "Third paragraph" in combined

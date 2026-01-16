Project: DocDistillery — a modular Python package to distill large documents and convert structured data (CSV) into narrative, with options for simple or advanced synthesis. The package must be usable as a CLI and as a library.

High-level goals:
- Accept PDF/DOCX/TXT/CSV inputs and produce high-quality summaries and narratives.
- Support three strategies: "sequential", "clustered", "insight_driven". Implement an "auto" selector that picks the strategy based on document size and redundancy.
- Use a Vector DB adapter (Qdrant in-memory by default) to store embeddings, deduplicate clusters, allow retrieval, and enable provenance.
- Allow optional LLM integration: local (e.g., huggingface/transformers model or pluggable local runtime) and cloud (OpenAI-like API). Implement an abstract LLMAdapter and two concrete examples (local+cloud). Provide clear extension points.
- Allow the user to choose output behavior:
  - Save to file in configurable format(s): Markdown (.md), plain text (.txt), JSON (.json), DOCX (.docx), PDF (.pdf) (PDF generation may use a light dependency like `markdown2` + `weasyprint` or make it optional).
  - OR return a single output string (no persistence).
- Always produce provenance metadata: which chunk(s) contributed to every part of the summary (chunk_id, score, page, excerpt), models used, timestamps, prompts used.
- Provide CLI commands and Python API; include unit tests, integration tests, and CI config for GitHub Actions.

Constraints and defaults:
- Language: English for code, but library must support multilingual text (embedding model default: "paraphrase-multilingual-MiniLM-L12-v2").
- Python: 3.10+ with type hints.
- Use sentence-transformers for embeddings; use transformers for summarizer (default mT5-small or a configurable abstractive model).
- Use qdrant-client and Qdrant in-memory collection for development. If Qdrant in-memory cannot be instantiated without a server in tests, provide a fall-back in-process adapter using Chroma or a pure-python KD-tree mock for tests.
- Unit tests must not perform network calls. Where external models or services are required, tests should mock them.
- Provide clear docstrings, README, and examples.
- The repo should include a `pyproject.toml` or `setup.cfg`, a `requirements.txt`, and a `tests/` folder.

Deliverables (for lead agent):
1. Repo skeleton listing (files + brief content description).
2. For each module below, produce code file(s) with full implementations + unit tests (pytest). Each file must have docstrings and type hints.
3. A top-level example script `examples/summarize_pdf.py` and `examples/csv2story.py`.
4. A working CLI skeleton `docdistillery/cli.py` using `click` or `argparse`.
5. README.md summarizing usage, strategy selection, output options, and guarantees (coverage_mode, audit_mode).
6. GitHub Actions workflow file for tests and linting.
MAI
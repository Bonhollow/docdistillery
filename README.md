![Logo](./images/logo.png)

# DocDistillery

DocDistillery is a modular Python package designed to distill large documents and transform structured data (CSV) into coherent narratives. It leverages vector embeddings, clustering, and LLMs to provide intelligent summarization and insight extraction.

## Quickstart

### Installation

Run the unified setup script to install system dependencies (macOS/Linux) and the Python environment:

```bash
git clone https://github.com/bonhollow/docdistillery.git
cd docdistillery
./setup.sh
```

This will:
- Install `pango`, `cairo`, and `libffi` (on macOS via Homebrew) for PDF export.
- Create a virtual environment (`.venv`).
- Install the package in editable mode.

### Basic Command

Summarize a PDF and see the result in your terminal:

```bash
docdistillery summarize --input report.pdf --no-save
```

---

## Hardware Requirements & Performance

DocDistillery supports a range of hardware configurations. Performance and model selection depend on available resources:

### Minimum Requirements
- **CPU**: Dual-core processor
- **RAM**: 8GB
- **Storage**: 2GB free space
- **Device**: Standard office laptop (e.g., MacBook Air, Dell XPS 13)

### Recommended for Local LLMs
- **CPU**: Quad-core or better (Apple M1/M2/M3 recommended)
- **RAM**: 16GB+
- **Model Selection**:
  - **8GB RAM**: Use `google/flan-t5-small` or Cloud LLMs
  - **16GB+ RAM**: Use `facebook/bart-large-cnn` (Default) or `pszemraj/led-base-book-summary` for long docs

> **Note**: First-time runs may take longer as models are downloaded and cached.

### Recommended Workflows

#### ⚡ Low Resource (8GB RAM / Intel Chips)
Use the `sequential` strategy and `standard` detail to conserve memory. Cloud LLMs are best here if available.
```bash
# Local (lightweight)
docdistillery summarize --input doc.pdf --strategy sequential --detail standard --no-save

# Cloud (best quality for low RAM)
export CLOUD_LLM_API_KEY="sk-..."
docdistillery summarize --input doc.pdf --llm cloud --model gpt-3.5-turbo
```

#### 🚀 High Performance (16GB+ RAM / Apple Silicon)
Leverage local power for maximum privacy and detail.
```bash
# High Detail Local Summarization (Private)
docdistillery summarize --input large_doc.pdf --detail detailed --format pdf --out summary.pdf

# Long Document Mode (using LED approximation via sliding window)
docdistillery summarize --input book.pdf --detail detailed --strategy sequential
```

#### ☁️ Cloud / API First (Any Device)
Best for users with OpenAI keys who want the highest possible quality without local resource usage.
```bash
# Summarize with GPT-4 (requires API key)
export CLOUD_LLM_API_KEY="sk-..."
docdistillery summarize --input complex_contract.pdf --llm cloud --model gpt-4 --detail detailed

# Insight extraction from CSV
docdistillery csv2story --input sales_data.csv --tone executive --out report.md
```

#### 🧪 Experimental / Custom Models
For users who want to try specific HuggingFace models.
```bash
# Use a specific summarization model
docdistillery summarize --input paper.pdf --model "philschmid/bart-large-cnn-samsum" --detail standard
```

---

## CLI Usage

DocDistillery provides a unified CLI for end-to-end processing.

### 1. Ingest
Extract text or view raw data from supported formats (`.pdf`, `.docx`, `.txt`, `.csv`).
```bash
docdistillery ingest --input data/source.pdf
```

### 2. Index
Chunk and store document embeddings in a vector database.
```bash
docdistillery index --doc-id "report_001" --input source.txt --db-type memory
```

### 3. Summarize
Run the full summarization pipeline.
```bash
docdistillery summarize --input long_doc.pdf --strategy auto --format pdf --out summary.pdf
```
- **Strategies**: `sequential`, `clustered`, `auto` (auto switches to clustered for files > 50KB).
- **Formats**: `md`, `txt`, `json`, `docx`, `pdf`.

### 4. CSV to Narrative
Extract insights from tabular data and build a structured story.
```bash
docdistillery csv2story --input metrics.csv --tone executive --out insights.md
```
- **Tones**: `executive`, `didactic`, `technical`.

---

## Library Usage

DocDistillery can be imported as a library for custom pipelines.

```python
from pipeline import ingest, chunk_pages, synthesize, export_summary

# 1. Ingest and Chunk
data = ingest("document.pdf")
chunks = chunk_pages(data["pages"], doc_id="my_doc")

# 2. Synthesize with default BasicSummarizer (or plug in an LLM)
summary = synthesize(chunks)

# 3. Export
md_text = export_summary(summary, format="md")
print(md_text)
```

---

## Core Concepts

### Summarization Strategies

- **Sequential** (Recommended): Processes chunks in document order using a sliding window. Best for coherence and stability.
- **Clustered**: Uses HDBSCAN/K-Means to group similar information. Best for highly redundant documents (requires `scikit-learn` and may be unstable on some systems).
- **Auto**: Switches between sequential and clustered based on document size.

### Guarantees & Auditing

DocDistillery provides tools to verify information coverage:
- **Provenance**: Every summary object includes a mapping of sections back to the source `chunk_ids`.
- **Audit Mode**: Compares the summary provenance against all document clusters to identify "blind spots".
- **Guarantees**: We guarantee that the `provenance` field accurately reflects the source chunks used. We do **not** guarantee that an LLM will perfectly capture every nuance, but the `audit_report` flags clusters that were entirely omitted.

### LLM Options & Privacy

DocDistillery is LLM-agnostic:
- **Local** (Default): Uses HuggingFace `transformers`.
  - default: `facebook/bart-large-cnn` (Stable, high quality, max 1024 tokens)
  - long docs: `pszemraj/led-base-book-summary` (Experimental, max 16k tokens)
- **Cloud**: Connects to OpenAI-compatible APIs. Requires `CLOUD_LLM_API_KEY`.
- **Privacy Warning**: Sending sensitive or proprietary documents to cloud LLMs may violate your data policy. Always prefer local models for sensitive data.

### Output Options

- **File Formats**: Supports persistent export to Markdown, JSON, Plain Text, DOCX, and PDF.
- **No-Save Mode**: Use the `--no-save` flag in CLI to return the formatted string directly to `stdout`, useful for piping to other tools.

---

## Development

### Running Tests
All tests are deterministic and do not require external network calls.
```bash
pytest tests/pipeline/
```

### Linting & Formatting
We use `ruff` for fast static analysis and formatting.
```bash
ruff format .
ruff check . --fix
```

### CI/CD
A GitHub Actions workflow (`.github/workflows/ci.yml`) is configured to run lints and tests across Python 3.10, 3.11, and 3.12 on every push.

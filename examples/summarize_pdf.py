"""
Example script: Programmatic PDF summarization using pipeline components.
This script demonstrates how to chain atomic components manually.
"""

from pipeline.chunk import chunk_pages
from pipeline.exporter import export_summary
from pipeline.ingest import ingest
from pipeline.summarize import synthesize


def run_example(pdf_path: str):
    print(f"--- Processing {pdf_path} ---")

    # 1. Ingest
    print("Ingesting PDF...")
    data = ingest(pdf_path)
    if "pages" not in data:
        print("Error: Input must be a text-based document.")
        return

    # 2. Chunking
    print("Chunking text...")
    chunks = chunk_pages(data["pages"], doc_id="example_doc")

    # 3. Summarization (Sequential Strategy)
    print("Synthesizing summary...")
    summary_obj = synthesize(chunks)

    # 4. Export to Markdown String
    print("Formatting output...")
    md_output = export_summary(summary_obj, format="md")

    print("\n--- Summary Result ---\n")
    print(md_output)


if __name__ == "__main__":
    # In a real scenario, you'd provide a path to an actual PDF.
    # For this example, we'll assume a file named 'sample.pdf' exists or
    # we'll catch the error if it doesn't.
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    try:
        run_example(path)
    except Exception as e:
        print(f"Note: Could not run example automatically: {e}")
        print("Usage: python examples/summarize_pdf.py path/to/your.pdf")

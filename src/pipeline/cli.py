import os
import sys

import click


@click.group()
def cli():
    """DocDistillery: End-to-end document processing and analytics."""
    pass


@click.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Path to input file.")
def ingest_cmd(input):
    """Extract text from a file."""
    from pipeline.ingest import ingest

    try:
        data = ingest(input)
        if "pages" in data:
            full_text = "\n".join([p["text"] for p in data["pages"]])
            click.echo(full_text)
        elif "data" in data:
            click.echo(data["data"].to_string())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


cli.add_command(ingest_cmd, name="ingest")


@click.command()
@click.option("--doc-id", "-d", required=True, help="Document ID.")
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Path to input file.")
@click.option("--db-type", type=click.Choice(["memory", "qdrant"]), default="memory", help="Vector DB type.")
def index(doc_id, input, db_type):
    """Chunk, embed, and store a document."""
    from pipeline.chunk import chunk_pages
    from pipeline.embed import Embedder
    from pipeline.ingest import ingest
    from pipeline.vectordb import InMemoryAdapter, QdrantAdapter

    try:
        data = ingest(input)
        if "pages" not in data:
            click.echo("Error: Indexing only supported for text/pdf/docx documents.", err=True)
            sys.exit(2)

        chunks = chunk_pages(data["pages"], doc_id=doc_id)

        embedder = Embedder()
        embeddings = embedder.embed_texts([c["text"] for c in chunks])

        db = InMemoryAdapter() if db_type == "memory" else QdrantAdapter()
        db.upsert_chunks("docs", chunks, embeddings)
        click.echo(f"Indexed {len(chunks)} chunks for doc {doc_id}.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@click.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Path to input file.")
@click.option("--mode", type=click.Choice(["simple", "advanced"]), default="simple")
@click.option("--strategy", type=click.Choice(["auto", "sequential", "clustered", "insight_driven"]), default="auto")
@click.option("--llm", type=click.Choice(["local", "cloud"]), default="local", help="LLM type for summarization.")
@click.option("--model", help="Specific LLM model name.")
@click.option("--out", "-o", type=click.Path(), help="Output file path.")
@click.option("--format", "-f", type=click.Choice(["md", "txt", "json", "docx", "pdf"]), default="md")
@click.option(
    "--detail",
    "-d",
    type=click.Choice(["brief", "standard", "detailed"]),
    default="standard",
    help="Summarization detail level.",
)
@click.option("--query", "-q", help="Query for insight-driven summarization.")
@click.option("--no-save", is_flag=True, help="Return output to stdout without saving.")
def summarize(input, mode, strategy, llm, model, out, format, detail, query, no_save):
    """Ingest, process, and summarize a document."""
    from pipeline.chunk import chunk_pages
    from pipeline.dedup import cluster_embeddings, select_representatives
    from pipeline.embed import Embedder
    from pipeline.exporter import export_summary
    from pipeline.ingest import ingest
    from pipeline.llm_adapter import CloudAdapter, LocalSummarizationAdapter
    from pipeline.retrieval import Retriever
    from pipeline.strategy import compute_doc_stats, select_strategy
    from pipeline.summarize import LLMSummarizer, synthesize

    try:
        data = ingest(input)
        doc_id = os.path.basename(input)

        if "pages" not in data:
            click.echo("Error: Summarization only supported for text/pdf/docx documents.", err=True)
            sys.exit(2)

        embedder = Embedder()

        if llm == "cloud":
            adapter = CloudAdapter(model_name=model or "gpt-3.5-turbo")
            if adapter.api_key == "PLACEHOLDER_KEY":
                click.echo("Warning: CLOUD_LLM_API_KEY not set. Falling back to local.", err=True)
                llm_obj = None
                use_cloud = False
            else:
                llm_obj = adapter
                use_cloud = True
        else:
            try:
                # Use BART model for stable summarization
                llm_obj = LocalSummarizationAdapter(model_name=model or "facebook/bart-large-cnn")
                use_cloud = False
            except ImportError:
                click.echo(
                    "Warning: 'transformers' not found. Using basic summarizer. "
                    "Install with 'pip install transformers torch'.",
                    err=True,
                )
                llm_obj = None
                use_cloud = False

        if use_cloud:
            pages_per_chunk = 15 if detail == "detailed" else (12 if detail == "standard" else 8)
            max_chunk_chars = 30000 if detail == "detailed" else (25000 if detail == "standard" else 15000)
        else:
            pages_per_chunk = 8 if detail == "detailed" else (6 if detail == "standard" else 4)
            max_chunk_chars = 12000 if detail == "detailed" else (10000 if detail == "standard" else 8000)

        summarizer = LLMSummarizer(llm_obj) if llm_obj else None

        chunks = chunk_pages(
            data["pages"], doc_id=doc_id, chunk_size_chars=max_chunk_chars, pages_per_chunk=pages_per_chunk
        )
        embeddings = embedder.embed_texts([c["text"] for c in chunks])

        stats = compute_doc_stats(chunks, embeddings)

        if strategy == "auto":
            strategy = select_strategy(stats, query=query)

        # chunking_config = recommend_chunking(strategy, len(data.get("pages", [])), stats)

        if strategy == "insight_driven" or strategy == "clustered":
            if len(chunks) > 1:
                labels = cluster_embeddings(embeddings)
                final_chunks = select_representatives(chunks, embeddings, labels)
            else:
                final_chunks = chunks
        else:
            final_chunks = chunks

        if strategy == "insight_driven" and query:
            from pipeline.vectordb import InMemoryAdapter

            db = InMemoryAdapter()
            db.upsert_chunks("temp", final_chunks, embedder.embed_texts([c["text"] for c in final_chunks]))

            retriever = Retriever(db, "temp", embedder)
            results = retriever.retrieve(query, top_k=12)

            if results:
                final_chunks = []
                for r in results:
                    if "payload" in r:
                        final_chunks.append(r["payload"])
                    elif "chunk_id" in r:
                        for c in chunks:
                            if c.get("chunk_id") == r["chunk_id"]:
                                final_chunks.append(c)
                                break

        new_embeddings = (
            embedder.embed_texts([c["text"] for c in final_chunks]) if len(final_chunks) != len(chunks) else embeddings
        )

        summary_obj = synthesize(
            final_chunks, summarizer=summarizer, llm=llm_obj, detail_level=detail, embeddings=new_embeddings
        )
        summary_obj["metadata"] = {"filename": doc_id, "strategy": strategy}
        summary_obj["doc_id"] = doc_id

        result = export_summary(summary_obj, out_path=out if not no_save else None, format=format)

        if no_save:
            click.echo(result)
        else:
            click.echo(f"Summary exported to {result}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@click.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Path to CSV file.")
@click.option("--tone", type=click.Choice(["executive", "didactic", "technical"]), default="executive")
@click.option("--out", "-o", type=click.Path(), help="Output file path.")
def csv2story(input, tone, out):
    """Extract insights from CSV and build a narrative story."""
    import pandas as pd

    from pipeline.csv2story import build_story
    from pipeline.csv_insights import extract_insights, insights_to_atomic_phrases

    try:
        df = pd.read_csv(input)
        insights = extract_insights(df)
        phrases = insights_to_atomic_phrases(insights)
        story = build_story(phrases, tone=tone)

        if out:
            with open(out, "w") as f:
                f.write(story)
            click.echo(f"Story exported to {out}")
        else:
            click.echo(story)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    cli()


cli.add_command(summarize)
cli.add_command(csv2story)


if __name__ == "__main__":
    main()

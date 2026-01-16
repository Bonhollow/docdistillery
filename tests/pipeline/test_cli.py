import os
import tempfile

import pandas as pd
from click.testing import CliRunner

from pipeline.cli import cli


def test_ingest_cli():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as tmp:
        tmp.write("Hello world from CLI.")
        tmp_path = tmp.name

    try:
        result = runner.invoke(cli, ["ingest", "-i", tmp_path])
        assert result.exit_code == 0
        assert "Hello world from CLI." in result.output
    finally:
        os.remove(tmp_path)


def test_csv2story_cli():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        result = runner.invoke(cli, ["csv2story", "-i", tmp_path, "--tone", "technical"])
        assert result.exit_code == 0
        assert "# Statistical Summary:" in result.output
    finally:
        os.remove(tmp_path)


def test_summarize_cli_no_save():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as tmp:
        tmp.write("This is a long document that needs summarization. " * 5)
        tmp_path = tmp.name

    try:
        result = runner.invoke(cli, ["summarize", "-i", tmp_path, "--no-save"])
        if result.exit_code != 0:
            print(result.output)
        assert result.exit_code == 0
        assert "# tmp" in result.output  # metadata uses filename
        assert "## TL;DR" in result.output
    finally:
        os.remove(tmp_path)


def test_cli_error_handling():
    runner = CliRunner()
    # Missing file
    result = runner.invoke(cli, ["ingest", "-i", "non_existent_file.txt"])
    assert result.exit_code == 2  # Click default for bad path

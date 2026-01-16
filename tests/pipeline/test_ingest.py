import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline.ingest import (
    extract_from_csv,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
    ingest,
)


def test_extract_text_from_txt():
    content = "Hello, this is a test text file."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = extract_text_from_txt(tmp_path)
        assert len(result) == 1
        assert result[0]["page"] == 1
        assert result[0]["text"] == content
    finally:
        os.remove(tmp_path)


def test_extract_from_csv():
    df_expected = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df_expected.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        result = extract_from_csv(tmp_path)
        pd.testing.assert_frame_equal(result, df_expected)
    finally:
        os.remove(tmp_path)


@patch("pdfplumber.open")
def test_extract_text_from_pdf(mock_pdf_open):
    # Mock pdfplumber structures
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF page content"
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__.return_value = mock_pdf
    mock_pdf_open.return_value = mock_pdf

    result = extract_text_from_pdf("dummy.pdf")
    assert len(result) == 1
    assert result[0]["text"] == "PDF page content"


@patch("docx.Document")
def test_extract_text_from_docx(mock_document):
    # Mock python-docx structures
    mock_para = MagicMock()
    mock_para.text = "Docx paragraph content"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para]
    mock_document.return_value = mock_doc

    result = extract_text_from_docx("dummy.docx")
    assert len(result) == 1
    assert result[0]["text"] == "Docx paragraph content"


def test_ingest_txt():
    content = "Ingest test content"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingest(tmp_path)
        assert "doc_id" in result
        assert result["pages"][0]["text"] == content
        assert result["metadata"]["extension"] == ".txt"
    finally:
        os.remove(tmp_path)


def test_ingest_csv():
    df_expected = pd.DataFrame({"a": [1]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df_expected.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        result = ingest(tmp_path)
        assert "doc_id" in result
        assert "data" in result
        pd.testing.assert_frame_equal(result["data"], df_expected)
    finally:
        os.remove(tmp_path)


def test_ingest_unsupported():
    with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="Unsupported file format"):
            ingest(tmp_path)
    finally:
        os.remove(tmp_path)


def test_ingest_not_found():
    with pytest.raises(FileNotFoundError):
        ingest("non_existent_file.txt")

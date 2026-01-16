import os
import re
import uuid
from typing import Any, Dict, List, Optional


def extract_text_from_pdf(path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a PDF file using pdfplumber.
    Handles various PDF layouts with multiple extraction strategies.

    Args:
        path (str): Path to the PDF file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing 'page' index and 'text'.
    """
    pages = []
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text = page.extract_text() or ""

                if not text or len(text.strip()) < 50:
                    alternative_methods = [
                        page.extract_text(x0=0, x1=page.width, y0=0, y1=page.height),
                        page.extract_text(x0=0, x1=page.width),
                        page.extract_text(),
                    ]
                    for alt in alternative_methods:
                        if alt and len(alt.strip()) > len(text.strip()):
                            text = alt

                if text:
                    text = _clean_extracted_text(text)
                    pages.append({"page": page_num, "text": text})
                else:
                    pages.append({"page": page_num, "text": ""})
    except Exception:
        pass
    return pages


def _clean_extracted_text(text: str) -> str:
    """
    Clean extracted text by removing common artifacts.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text with artifacts removed.
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if len(line) <= 5 and line.replace(".", "").replace("-", "").isdigit():
            continue

        if re.match(r"^(page\s*\d*|\d+\s*$)", line, re.IGNORECASE):
            continue

        if re.match(r"^\d+$", line):
            continue

        line = re.sub(r"\s+", " ", line)
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def extract_text_from_docx(path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a DOCX file using python-docx.
    Treats the entire document as a single "page" for now.

    Args:
        path (str): Path to the DOCX file.

    Returns:
        List[Dict[str, Any]]: A list containing a single dictionary with 'page' 1 and the full text.
    """
    try:
        from docx import Document

        doc = Document(path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        text = "\n".join(full_text)
        return [{"page": 1, "text": text}]
    except Exception:
        return []


def extract_text_from_txt(path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a TXT file.

    Args:
        path (str): Path to the TXT file.

    Returns:
        List[Dict[str, Any]]: A list containing a single dictionary with 'page' 1 and the full text.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return [{"page": 1, "text": text}]
    except Exception:
        return []


def extract_from_csv(path: str) -> "pd.DataFrame":
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        import pandas as pd

        return pd.read_csv(path)
    except Exception:
        import pandas as pd

        return pd.DataFrame()


def ingest(path: str) -> Dict[str, Any]:
    """
    Main entry point for ingesting documents. Detects format and routes to appropriate extractor.

    Args:
        path (str): Path to the file.

    Returns:
        Dict[str, Any]: A dictionary containing 'doc_id', 'pages' (or 'data' for CSV), and 'metadata'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    doc_id = str(uuid.uuid4())
    metadata = {"filename": os.path.basename(path), "extension": ext, "path": path}

    if ext == ".pdf":
        pages = extract_text_from_pdf(path)
        return {"doc_id": doc_id, "pages": pages, "metadata": metadata}
    elif ext == ".docx":
        pages = extract_text_from_docx(path)
        return {"doc_id": doc_id, "pages": pages, "metadata": metadata}
    elif ext == ".txt":
        pages = extract_text_from_txt(path)
        return {"doc_id": doc_id, "pages": pages, "metadata": metadata}
    elif ext == ".csv":
        data = extract_from_csv(path)
        return {"doc_id": doc_id, "data": data, "metadata": metadata}
    else:
        raise ValueError(f"Unsupported file format: {ext}")

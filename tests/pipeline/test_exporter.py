import json
import os
import tempfile

import pytest

from pipeline.exporter import export_summary


@pytest.fixture
def sample_summary():
    return {
        "doc_id": "test-123",
        "tldr": "Quick summary",
        "executive": ["High efficiency", "Cost reduction"],
        "sections": [{"title": "Detail 1", "content": "Full content of section 1."}],
        "metadata": {"filename": "TestReport.txt"},
        "provenance": {"chunk_1": [0]},
    }


def test_export_json_string(sample_summary):
    result = export_summary(sample_summary, format="json")
    data = json.loads(result)
    assert data["doc_id"] == "test-123"


def test_export_md_string(sample_summary):
    result = export_summary(sample_summary, format="md")
    assert "# TestReport.txt" in result
    assert "## TL;DR" in result
    assert "Quick summary" in result
    assert "- High efficiency" in result


def test_export_to_file(sample_summary):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "report.md")
        result_path = export_summary(sample_summary, out_path=out_path, format="md")

        assert os.path.exists(out_path)
        assert result_path == out_path

        # Check content
        with open(out_path, "r") as f:
            content = f.read()
            assert "# TestReport.txt" in content

        # Check provenance
        prov_path = os.path.join(tmpdir, "provenance_test-123.json")
        assert os.path.exists(prov_path)
        with open(prov_path, "r") as f:
            prov_data = json.load(f)
            assert prov_data["chunk_1"] == [0]


def test_export_binary_no_path(sample_summary):
    with pytest.raises(ValueError, match="requires out_path"):
        export_summary(sample_summary, format="docx")


def test_export_unsupported_format(sample_summary):
    with pytest.raises(ValueError, match="Unsupported format"):
        export_summary(sample_summary, format="png")

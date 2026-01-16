import os
from unittest.mock import MagicMock, patch

import pytest

from pipeline.llm_adapter import CloudAdapter, LLMAdapter


class MockLocalAdapter(LLMAdapter):
    """Deterministic mock for LocalTransformersAdapter testing."""

    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        return f"Local Echo: {prompt[:10]}"


def test_mock_local_adapter():
    adapter = MockLocalAdapter()
    result = adapter.generate("Hello world", 10)
    assert result == "Local Echo: Hello worl"


@patch("requests.post")
def test_cloud_adapter_payload(mock_post):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "Cloud response"}}]}
    mock_post.return_value = mock_response

    os.environ["CLOUD_LLM_API_KEY"] = "test_key"
    adapter = CloudAdapter(api_url="http://mock-api.com", model_name="test-model")

    result = adapter.generate("Test prompt", max_tokens=50, temperature=0.7)

    assert result == "Cloud response"

    # Verify payload
    args, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["model"] == "test-model"
    assert payload["max_tokens"] == 50
    assert payload["temperature"] == 0.7
    assert payload["messages"][0]["content"] == "Test prompt"
    assert kwargs["headers"]["Authorization"] == "Bearer test_key"


@patch("requests.post")
def test_cloud_adapter_retry(mock_post):
    # Setup mock response to fail twice then succeed
    fail_response = MagicMock()
    fail_response.status_code = 500

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {"choices": [{"message": {"content": "Success after retry"}}]}

    mock_post.side_effect = [fail_response, fail_response, success_response]

    # Patch time.sleep to speed up tests
    with patch("time.sleep"):
        adapter = CloudAdapter()
        result = adapter.generate("test", 10)
        assert result == "Success after retry"
        assert mock_post.call_count == 3


@patch("requests.post")
def test_cloud_adapter_fail(mock_post):
    import requests

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Unauthorized")
    mock_post.return_value = mock_response

    adapter = CloudAdapter()
    with pytest.raises(RuntimeError, match="Cloud LLM request failed"):
        adapter.generate("test", 10)

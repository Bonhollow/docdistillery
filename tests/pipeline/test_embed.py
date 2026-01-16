import hashlib
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np

from pipeline.embed import Embedder


class MockEmbedder:
    """
    Lightweight mock embedder for testing other modules.
    Produces deterministic vectors based on text hashing.
    """

    def __init__(self, dimension: int = 384, normalize: bool = True):
        self.dimension = dimension
        self.normalize = normalize

    def _hash_text(self, text: str) -> np.ndarray:
        # Create a deterministic vector based on SHA256 hash of text
        h = hashlib.sha256(text.encode()).digest()
        # Convert bytes to uint8 then to float
        vec = np.frombuffer(h, dtype=np.uint8).astype(float)

        # Expand or crop to the desired dimension
        if len(vec) < self.dimension:
            # Repeat the hash vector until it covers the dimension
            vec = np.tile(vec, (self.dimension // len(vec)) + 1)[: self.dimension]
        else:
            vec = vec[: self.dimension]

        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return np.array([self._hash_text(t) for t in texts])

    def embed_text(self, text: str) -> np.ndarray:
        return self._hash_text(text)


@patch("sentence_transformers.SentenceTransformer")
def test_embedder_interface(mock_st_class):
    # Setup mock ST model
    mock_st_instance = MagicMock()
    mock_st_class.return_value = mock_st_instance

    # Mock return value for encode
    dummy_embedding = np.random.rand(384).astype(np.float32)
    mock_st_instance.encode.return_value = dummy_embedding

    embedder = Embedder(model_name="dummy-model", normalize=True)

    # Test single text embedding
    result = embedder.embed_text("hello world")

    # Verify ST was initialized correctly
    mock_st_class.assert_called_once_with("dummy-model", device="cpu")

    # Verify encode was called with correct parameters
    mock_st_instance.encode.assert_called_once_with("hello world", convert_to_numpy=True, normalize_embeddings=True)
    assert np.array_equal(result, dummy_embedding)


@patch("sentence_transformers.SentenceTransformer")
def test_embedder_batch_interface(mock_st_class):
    mock_st_instance = MagicMock()
    mock_st_class.return_value = mock_st_instance

    dummy_embeddings = np.random.rand(2, 384).astype(np.float32)
    mock_st_instance.encode.return_value = dummy_embeddings

    embedder = Embedder(normalize=False)
    texts = ["one", "two"]
    result = embedder.embed_texts(texts, batch_size=10)

    mock_st_instance.encode.assert_called_once_with(
        texts, batch_size=10, convert_to_numpy=True, normalize_embeddings=False
    )
    assert np.array_equal(result, dummy_embeddings)


def test_mock_embedder_determinism():
    mock = MockEmbedder(dimension=128, normalize=True)
    text = "consistent text"

    v1 = mock.embed_text(text)
    v2 = mock.embed_text(text)

    assert np.allclose(v1, v2)
    assert v1.shape == (128,)
    # Verify normalization
    assert np.allclose(np.linalg.norm(v1), 1.0)


def test_mock_embedder_batch():
    mock = MockEmbedder(dimension=64, normalize=False)
    texts = ["a", "b", "c"]

    embeddings = mock.embed_texts(texts)

    assert embeddings.shape == (3, 64)
    assert not np.allclose(embeddings[0], embeddings[1])

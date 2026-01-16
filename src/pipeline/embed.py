from typing import List, Optional


class Embedder:
    """
    Wrapper for sentence-transformers models to produce text embeddings.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        normalize: bool = True,
    ):
        """
        Initializes the Embedder.

        Args:
            model_name (str): Name of the sentence-transformers model.
            device (str): Device to use (cpu, cuda, mps).
            normalize (bool): Whether to L2-normalize embeddings.
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> "np.ndarray":
        """
        Converts a list of strings into a matrix of embeddings.

        Args:
            texts (List[str]): Input strings.
            batch_size (int): Size of processing batches.

        Returns:
            np.ndarray: (N, D) matrix of embeddings.
        """
        import numpy as np

        if not texts:
            return np.array([])

        # sentence-transformers encode method supports normalize_embeddings directly
        embeddings = self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=self.normalize
        )
        return embeddings

    def embed_text(self, text: str) -> "np.ndarray":
        """
        Encodes a single text into an embedding.

        Args:
            text (str): String to embed.

        Returns:
            np.ndarray: Array of shape (D,).
        """
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=self.normalize)
        return embedding

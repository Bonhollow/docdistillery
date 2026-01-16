import os
import time
from abc import ABC, abstractmethod

import requests


class LLMAdapter(ABC):
    """
    Abstract interface for Large Language Models.
    All implementations must be stateless and thread-safe.

    Warning: Do not send sensitive data to external cloud models unless permitted.
    """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        """
        Generates text from a prompt.

        Args:
            prompt (str): Input text.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature (0.0 = deterministic).

        Returns:
            str: Generated text.
        """
        pass


class LocalTransformersAdapter(LLMAdapter):
    """
    Adapter for local models using the HuggingFace transformers library.
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: str = "cpu"):
        # Lazy import to avoid heavy dependency if not used
        from transformers import pipeline

        self.pipeline = pipeline("text2text-generation", model=model_name, device=device)

    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        # Note: transformers pipeline parameters vary by task.
        # For text2text-generation, we use max_new_tokens.
        # We also enforce truncation to the model's max length to avoid indexing errors.
        
        # Access tokenizer from the pipeline to get max length
        tokenizer = self.pipeline.tokenizer
        max_length = getattr(tokenizer, "model_max_length", 1024)
        if max_length > 10**6: # Some models have a placeholder huge value
            max_length = 1024
            
        results = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            truncation=True,
            max_length=max_length
        )
        return results[0]["generated_text"]


class CloudAdapter(LLMAdapter):
    """
    Adapter for cloud LLMs (OpenAI-compatible REST API).
    Expects CLOUD_LLM_API_KEY environment variable.
    """

    def __init__(self, api_url: str = "https://api.openai.com/v1/chat/completions", model_name: str = "gpt-3.5-turbo"):
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = os.environ.get("CLOUD_LLM_API_KEY", "PLACEHOLDER_KEY")

    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    # Simplified OpenAI-style response parsing
                    return data["choices"][0]["message"]["content"].strip()

                if 500 <= response.status_code < 600:
                    # Retry on server errors
                    time.sleep(2**attempt)
                    continue

                response.raise_for_status()

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Cloud LLM request failed after {max_retries} attempts: {e}") from e
                time.sleep(2**attempt)

        raise RuntimeError("Cloud LLM request failed: unknown error")


# Example usage with Summarizer:
# summarizer = synthesize(chunks, summarizer=my_sum, llm=LocalTransformersAdapter())
# result = summarizer.generate("Summarize this...", max_tokens=100)

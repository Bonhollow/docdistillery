import os
import time
from abc import ABC, abstractmethod

import requests


class LLMAdapter(ABC):
    """
    Abstract interface for Large Language Models.
    """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        pass


class LocalSummarizationAdapter(LLMAdapter):
    """
    Adapter for local summarization using HuggingFace transformers.
    Uses BART-large-cnn by default (stable, high quality).
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: str = "cpu"):
        from transformers import pipeline

        self.model_name = model_name
        self.device = device
        
        # Use summarization pipeline
        self.pipeline = pipeline(
            "summarization",
            model=model_name,
            device=device,
            truncation=True,
        )
        # BART handles up to 1024 tokens
        self.max_input_length = 1024

    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        """
        Generate summary from input text.
        For summarization pipeline, 'prompt' is treated as the text to summarize.
        """
        # Ensure we don't exceed model limits
        input_length = len(prompt.split())
        
        # Adjust min/max length based on input
        min_length = min(50, max(10, input_length // 10))
        max_length = min(max_tokens, max(100, input_length // 3))
        
        try:
            results = self.pipeline(
                prompt,
                max_length=max_length,
                min_length=min_length,
                do_sample=(temperature > 0),
                truncation=True,
            )
            return results[0]["summary_text"]
        except Exception as e:
            # If summarization fails, return truncated input
            return prompt[:max_tokens * 4] + "..."


# Backwards compatibility alias
LocalTransformersAdapter = LocalSummarizationAdapter


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

        # For summarization, wrap in a system prompt
        system_prompt = "You are a professional summarizer. Create comprehensive, well-structured summaries that preserve key details and maintain document order."
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize the following text:\n\n{prompt}"}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()

                if 500 <= response.status_code < 600:
                    time.sleep(2**attempt)
                    continue

                response.raise_for_status()

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Cloud LLM request failed after {max_retries} attempts: {e}") from e
                time.sleep(2**attempt)

        raise RuntimeError("Cloud LLM request failed: unknown error")

"""
Azure OpenAI Embeddings Provider

This module provides async embeddings generation using Azure OpenAI's text-embedding-3-large model.
"""

import asyncio
import os
from typing import List, Optional, Union
from openai import AsyncAzureOpenAI
import numpy as np
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for Azure OpenAI embeddings"""
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "https://autogen-testing-ubs.cognitiveservices.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15")
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    deployment_name: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    model: str = "text-embedding-3-large"
    dimensions: int = 3072  # Default dimensions for text-embedding-3-large
    batch_size: int = 100  # Azure OpenAI has limits on batch size


class AsyncAzureOpenAIEmbeddings:
    """
    Async Azure OpenAI embeddings provider for generating dense vectors.
    Uses text-embedding-3-large model for high-quality semantic embeddings.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the Azure OpenAI embeddings provider.

        Args:
            config: Optional configuration object. If not provided, uses environment variables.
        """
        self.config = config or EmbeddingConfig()

        # Validate configuration
        if not self.config.azure_endpoint or not self.config.api_key:
            raise ValueError(
                "Azure OpenAI credentials not configured. Please set AZURE_OPENAI_ENDPOINT "
                "and AZURE_OPENAI_API_KEY environment variables or provide configuration."
            )

        # Initialize async client
        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.config.azure_endpoint,
            api_key=self.config.api_key,
            api_version=self.config.api_version
        )

    async def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text using Azure OpenAI.

        Args:
            text: Single text string or list of texts to embed

        Returns:
            numpy array of embeddings. Shape (dimensions,) for single text,
            or (num_texts, dimensions) for multiple texts.
        """
        # Convert single string to list for uniform processing
        texts = [text] if isinstance(text, str) else text
        is_single = isinstance(text, str)

        # Process in batches if needed
        all_embeddings = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            try:
                response = await self.client.embeddings.create(
                    model=self.config.deployment_name,
                    input=batch,
                    dimensions=self.config.dimensions
                )

                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Return single vector if input was single string
        return embeddings_array[0] if is_single else embeddings_array

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            documents: List of document texts to embed

        Returns:
            List of embedding vectors (as lists of floats)
        """
        embeddings = await self.embed_text(documents)
        return embeddings.tolist()

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = await self.embed_text(query)
        return embedding.tolist()

    @property
    def embedding_dimensions(self) -> int:
        """Return the dimensionality of the embeddings"""
        return self.config.dimensions

    async def close(self):
        """Close the async client"""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Convenience function for quick embedding generation
async def generate_embeddings(
    texts: Union[str, List[str]],
    config: Optional[EmbeddingConfig] = None
) -> np.ndarray:
    """
    Convenience function to generate embeddings without managing client lifecycle.

    Args:
        texts: Text or list of texts to embed
        config: Optional configuration

    Returns:
        numpy array of embeddings
    """
    async with AsyncAzureOpenAIEmbeddings(config) as embedder:
        return await embedder.embed_text(texts)
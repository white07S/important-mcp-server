"""
Embeddings Provider Module

Provides embedding generation capabilities for vector search.
"""

from .azure_openai import (
    AsyncAzureOpenAIEmbeddings,
    EmbeddingConfig,
    generate_embeddings
)

__all__ = [
    'AsyncAzureOpenAIEmbeddings',
    'EmbeddingConfig',
    'generate_embeddings'
]
"""
Qdrant Document Ingestion Module

This module handles ingesting documents from chunks.json into Qdrant with:
- Dense embeddings for semantic search
- Sparse embeddings for keyword/full-text search
- Metadata filtering support
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    TextIndexParams,
    TokenizerType
)
import numpy as np
from tqdm.asyncio import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from provider.embeddings import AsyncAzureOpenAIEmbeddings, EmbeddingConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for document ingestion"""
    # Qdrant configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "documents"

    # File paths
    chunks_file: str = "/Users/preetam/Develop/mcp_servers/doc-mcp/parser/chunks.json"
    metadata_file: str = "/Users/preetam/Develop/mcp_servers/doc-mcp/parser/chunks_meta_data.json"

    # Processing configuration
    batch_size: int = 50  # Number of documents to process at once
    recreate_collection: bool = False  # Whether to recreate collection if it exists

    # Vector configuration
    dense_vector_size: int = 3072  # text-embedding-3-large dimensions

    # Full-text search configuration
    enable_text_index: bool = True  # Enable full-text search indexing
    text_index_tokenizer: str = "word"  # Tokenizer for FTS: "word", "multilingual", "whitespace"


class DocumentIngester:
    """
    Handles document ingestion into Qdrant with hybrid search capabilities.
    """

    @staticmethod
    def _resolve_vectors_count(collection_info) -> Optional[int]:
        """
        Qdrant renamed `vectors_count` to `indexed_vectors_count` in newer releases.
        This helper keeps compatibility with both versions.
        """
        vectors_count = getattr(collection_info, "vectors_count", None)
        if vectors_count is None:
            vectors_count = getattr(collection_info, "indexed_vectors_count", None)
        return vectors_count

    def __init__(self, config: Optional[IngestionConfig] = None):
        """
        Initialize the document ingester.

        Args:
            config: Optional ingestion configuration
        """
        self.config = config or IngestionConfig()

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port
        )

        # Initialize embedding provider (only dense embeddings from OpenAI)
        self.dense_embedder = AsyncAzureOpenAIEmbeddings()

    def _load_data(self) -> tuple[List[Dict], Dict]:
        """
        Load chunks and metadata from JSON files.

        Returns:
            Tuple of (chunks list, metadata dict)
        """
        # Load chunks
        chunks_path = Path(self.config.chunks_file)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # Load metadata
        metadata_path = Path(self.config.metadata_file)
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")

        logger.info(f"Loaded {len(chunks)} chunks and metadata")
        return chunks, metadata

    async def _create_collection(self) -> None:
        """
        Create or recreate the Qdrant collection with appropriate configuration.
        """
        collection_exists = self.qdrant_client.collection_exists(self.config.collection_name)

        if collection_exists:
            if self.config.recreate_collection:
                logger.info(f"Deleting existing collection: {self.config.collection_name}")
                self.qdrant_client.delete_collection(self.config.collection_name)
            else:
                logger.info(f"Collection {self.config.collection_name} already exists")
                return

        # Configure dense vectors for semantic search
        vectors_config = {
            "dense": VectorParams(
                size=self.config.dense_vector_size,
                distance=Distance.COSINE
            )
        }

        # Create collection
        logger.info(f"Creating collection: {self.config.collection_name}")
        self.qdrant_client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=vectors_config
        )

        # Configure text indexing for full-text search if enabled
        if self.config.enable_text_index:
            logger.info("Configuring text indexes for full-text search...")

            # Create text index for the searchable_text field
            self.qdrant_client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="searchable_text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD if self.config.text_index_tokenizer == "word" else TokenizerType.MULTILINGUAL,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True
                )
            )

            # Also index the text field for filtering
            self.qdrant_client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD if self.config.text_index_tokenizer == "word" else TokenizerType.MULTILINGUAL,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True
                )
            )

        logger.info("Collection created successfully with text indexing")


    def _create_searchable_text(self, chunk: Dict) -> str:
        """
        Create a combined searchable text from chunk components.

        Args:
            chunk: Chunk dictionary

        Returns:
            Combined text for embedding generation
        """
        components = []

        # Add section breadcrumb if available
        if chunk.get('section_breadcrumb'):
            breadcrumb_text = " > ".join(chunk['section_breadcrumb'])
            components.append(f"Section: {breadcrumb_text}")

        # Add section title if available and not in breadcrumb
        if chunk.get('section_title'):
            components.append(f"Title: {chunk['section_title']}")

        # Add the main text
        components.append(chunk['text'])

        # Join with newlines to maintain structure
        return "\n".join(components)

    async def _process_batch(
        self,
        chunks: List[Dict],
        global_metadata: Dict,
        start_idx: int
    ) -> List[PointStruct]:
        """
        Process a batch of chunks into Qdrant points.

        Args:
            chunks: List of chunk dictionaries
            global_metadata: Global document metadata
            start_idx: Starting index for this batch

        Returns:
            List of PointStruct objects ready for insertion
        """
        # Create combined texts for embedding (includes title, breadcrumb, and text)
        combined_texts = [self._create_searchable_text(chunk) for chunk in chunks]

        # Generate dense embeddings on the combined text
        logger.debug(f"Generating dense embeddings for {len(combined_texts)} combined texts")
        dense_embeddings = await self.dense_embedder.embed_documents(combined_texts)

        # Create points
        points = []
        for i, chunk in enumerate(chunks):
            # Create a unique integer ID from the chunk ID
            # Use hash to convert string ID to integer
            point_id = int(hashlib.md5(chunk['id'].encode()).hexdigest()[:8], 16)

            # Prepare payload with chunk metadata and global metadata
            payload = {
                # Chunk-specific metadata
                "chunk_id": chunk['id'],
                "doc_id": chunk.get('doc_id', ''),
                "text": chunk['text'],
                "searchable_text": combined_texts[i],  # Combined text for FTS
                "section_title": chunk.get('section_title', ''),
                "section_level": chunk.get('section_level', 0),
                "section_breadcrumb": chunk.get('section_breadcrumb', []),

                # Global document metadata (if available)
                "doc_name": global_metadata.get('doc_name', ''),
                "file_path": global_metadata.get('file_path', ''),
                "parsed_file_path": global_metadata.get('parsed_file_path', ''),
                "taxonomy": global_metadata.get('taxonomy', ''),
                "last_modified": global_metadata.get('last_modified', ''),
                "title": global_metadata.get('title', ''),
                "summary": global_metadata.get('summary', ''),
                "origin": global_metadata.get('origin', ''),
                "owning_division": global_metadata.get('owning_division', ''),

                # Additional indexing metadata
                "chunk_index": start_idx + i,
                "text_length": len(chunk['text']),
                "has_section": bool(chunk.get('section_title'))
            }

            # Create vectors dictionary (only dense vectors)
            vectors = {
                "dense": dense_embeddings[i]
            }

            # Create point
            point = PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload
            )

            points.append(point)

        return points

    async def ingest_documents(self) -> Dict[str, Any]:
        """
        Main ingestion method to load and index documents into Qdrant.

        Returns:
            Dictionary with ingestion statistics
        """
        try:
            # Load data
            chunks, metadata = self._load_data()

            # Create collection
            await self._create_collection()

            # Process chunks in batches
            total_chunks = len(chunks)
            total_points = 0
            failed_batches = 0

            logger.info(f"Starting ingestion of {total_chunks} chunks")

            # Use async progress bar
            with tqdm(total=total_chunks, desc="Ingesting chunks") as pbar:
                for i in range(0, total_chunks, self.config.batch_size):
                    batch = chunks[i:i + self.config.batch_size]

                    try:
                        # Process batch
                        points = await self._process_batch(batch, metadata, i)

                        # Upsert to Qdrant
                        self.qdrant_client.upsert(
                            collection_name=self.config.collection_name,
                            points=points
                        )

                        total_points += len(points)
                        pbar.update(len(batch))

                    except Exception as e:
                        logger.error(f"Failed to process batch {i//self.config.batch_size}: {e}")
                        failed_batches += 1
                        pbar.update(len(batch))

            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.config.collection_name)

            # Prepare results
            results = {
                "status": "success" if failed_batches == 0 else "partial",
                "total_chunks": total_chunks,
                "total_points_indexed": total_points,
                "failed_batches": failed_batches,
                "collection_name": self.config.collection_name,
                "vectors_count": self._resolve_vectors_count(collection_info),
                "points_count": collection_info.points_count,
                "indexed_vectors": list(collection_info.config.params.vectors.keys()),
                "text_index_enabled": self.config.enable_text_index,
                "metadata_fields": list(metadata.keys()) if metadata else []
            }

            logger.info(f"Ingestion completed: {results}")
            return results

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise

        finally:
            # Clean up
            await self.dense_embedder.close()

    async def verify_ingestion(self) -> Dict[str, Any]:
        """
        Verify that documents were properly ingested.

        Returns:
            Verification statistics
        """
        try:
            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.config.collection_name)

            # Sample a few points to verify
            sample_points = self.qdrant_client.retrieve(
                collection_name=self.config.collection_name,
                ids=[1, 2, 3],  # Sample IDs
                with_vectors=False,
                with_payload=True
            )

            # Check for required fields
            required_fields = ["chunk_id", "doc_id", "text"]
            sample_payloads = []

            for point in sample_points:
                if point.payload:
                    missing_fields = [f for f in required_fields if f not in point.payload]
                    sample_payloads.append({
                        "id": point.id,
                        "has_required_fields": len(missing_fields) == 0,
                        "missing_fields": missing_fields,
                        "payload_keys": list(point.payload.keys())
                    })

            return {
                "collection_exists": True,
                "points_count": collection_info.points_count,
                "vectors_count": self._resolve_vectors_count(collection_info),
                "indexed_vectors_config": collection_info.config.params.vectors,
                "text_index_enabled": self.config.enable_text_index,
                "sample_points": sample_payloads
            }

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                "collection_exists": False,
                "error": str(e)
            }


async def main():
    """
    Main function to run document ingestion.
    """
    # Configure ingestion
    config = IngestionConfig(
        recreate_collection=True,  # Start fresh
        batch_size=50,
        enable_text_index=True  # Enable full-text search
    )

    # Create ingester
    ingester = DocumentIngester(config)

    # Run ingestion
    logger.info("Starting document ingestion...")
    results = await ingester.ingest_documents()

    # Verify ingestion
    logger.info("Verifying ingestion...")
    verification = await ingester.verify_ingestion()

    # Print results
    print("\n=== Ingestion Results ===")
    print(json.dumps(results, indent=2))
    print("\n=== Verification Results ===")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

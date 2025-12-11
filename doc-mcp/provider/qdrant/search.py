"""
Qdrant Hybrid Search Module

This module provides hybrid search capabilities combining:
- Dense vector similarity search (semantic search)
- Sparse vector search (keyword/full-text search)
- Metadata filtering
- Result fusion and reranking
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Literal, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
    MatchAny,
    Range
)
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from provider.embeddings import AsyncAzureOpenAIEmbeddings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search mode enumeration"""
    DENSE = "dense"  # Semantic search only
    FTS = "fts"  # Full-text search only
    HYBRID = "hybrid"  # Combined FTS + semantic search
    FUSION = "fusion"  # Advanced fusion with reranking


@dataclass
class SearchConfig:
    """Configuration for search operations"""
    # Qdrant configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "documents"
    metadata_paths: Optional[List[str]] = None  # Optional metadata files to enrich grouped results

    # Search configuration
    search_mode: SearchMode = SearchMode.HYBRID
    top_k: int = 10  # Number of results to return
    prefetch_limit: int = 100  # Number of candidates for reranking

    # Fusion configuration
    fusion_method: Literal["rrf", "dbsf"] = "rrf"  # Reciprocal Rank Fusion or Distribution-Based Score Fusion
    fts_weight: float = 0.3  # Weight for full-text search in hybrid mode
    dense_weight: float = 0.7  # Weight for dense vectors in hybrid mode

    # Reranking configuration
    enable_reranking: bool = True  # Enable reranking in hybrid/fusion modes
    rerank_with: Literal["dense", "fts", "cross"] = "dense"  # What to use for final reranking

    # Search parameters
    enable_metadata_filter: bool = True
    return_full_text: bool = True  # Whether to return full text in results
    include_search_fields: bool = True  # Include info about what fields were searched


@dataclass
class SearchFilter:
    """Search filter configuration for metadata filtering"""
    doc_id: Optional[str] = None
    doc_name: Optional[str] = None
    taxonomy: Optional[str] = None
    origin: Optional[str] = None
    owning_division: Optional[str] = None
    section_level: Optional[int] = None
    has_section: Optional[bool] = None
    text_contains: Optional[str] = None  # Full-text filter on the text field
    custom_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Individual search result"""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    section_title: Optional[str] = None
    section_breadcrumb: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    match_type: str = "hybrid"  # dense, fts, or hybrid


DOC_METADATA_FIELDS = [
    "doc_name",
    "file_path",
    "parsed_file_path",
    "taxonomy",
    "last_modified",
    "title",
    "summary",
    "origin",
    "owning_division",
    "owning_divison"  # Backwards compatibility with legacy metadata typo
]


class HybridSearcher:
    """
    Implements hybrid search combining dense embeddings and full-text search with metadata filtering.
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize the hybrid searcher.

        Args:
            config: Optional search configuration
        """
        self.config = config or SearchConfig()

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port
        )

        # Initialize embedding provider (only dense embeddings from OpenAI)
        self.dense_embedder = AsyncAzureOpenAIEmbeddings()
        self.metadata_store = self._load_metadata_store(self._resolve_metadata_paths())

    def _resolve_metadata_paths(self) -> List[Path]:
        """
        Determine which metadata JSON files should be loaded.
        Priority:
        1. Explicit paths from config.metadata_paths
        2. Default parser/chunks_meta_data.json if it exists
        """
        raw_paths: List[Union[str, Path]] = []
        if self.config.metadata_paths:
            if isinstance(self.config.metadata_paths, (str, Path)):
                raw_paths = [self.config.metadata_paths]
            elif isinstance(self.config.metadata_paths, Iterable):
                raw_paths = list(self.config.metadata_paths)
            else:
                raw_paths = [self.config.metadata_paths]

        resolved_paths: List[Path] = []
        sources = raw_paths

        if not sources:
            try:
                repo_root = Path(__file__).resolve().parents[2]
            except IndexError:
                repo_root = Path(__file__).resolve().parent
            default_path = repo_root / "parser" / "chunks_meta_data.json"
            if default_path.exists():
                sources = [default_path]
            else:
                sources = []

        for source in sources:
            path = Path(source).expanduser() if not isinstance(source, Path) else source.expanduser()
            if path.exists():
                resolved_paths.append(path)
            else:
                logger.warning(f"Metadata file not found: {path}")

        return resolved_paths

    def _load_metadata_store(self, paths: List[Path]) -> Dict[str, Dict[str, Any]]:
        """
        Load metadata records from the provided JSON files.
        """
        store: Dict[str, Dict[str, Any]] = {}
        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                normalized = self._normalize_metadata_payload(payload)
                if normalized:
                    store.update(normalized)
                    unique_docs = {
                        value.get("doc_id")
                        for value in normalized.values()
                        if value.get("doc_id")
                    }
                    doc_count = len(unique_docs)
                    logger.info(f"Loaded {doc_count} metadata entries from {path}")
                else:
                    logger.warning(f"No document metadata found in {path}")
            except FileNotFoundError:
                logger.warning(f"Metadata file not found: {path}")
            except Exception as exc:
                logger.warning(f"Failed to load metadata from {path}: {exc}")

        return store

    @staticmethod
    def _normalize_metadata_payload(payload: Any) -> Dict[str, Dict[str, Any]]:
        """
        Normalize various metadata JSON formats into a flat mapping keyed by doc_id/doc_name.
        """
        records: List[Dict[str, Any]] = []

        if isinstance(payload, list):
            records = [record for record in payload if isinstance(record, dict)]
        elif isinstance(payload, dict):
            if all(isinstance(value, dict) for value in payload.values()):
                for key, value in payload.items():
                    record = dict(value)
                    if key and "doc_id" not in record:
                        record.setdefault("doc_id", key)
                    records.append(record)
            else:
                records = [payload]
        else:
            return {}

        store: Dict[str, Dict[str, Any]] = {}
        for record in records:
            doc_id = record.get("doc_id") or record.get("docId")
            doc_name = record.get("doc_name") or record.get("docName")
            identifier = doc_id or doc_name
            if not identifier:
                continue

            normalized = dict(record)
            normalized["doc_id"] = doc_id or doc_name or identifier
            normalized.setdefault("doc_name", doc_name or doc_id or identifier)

            store[normalized["doc_id"]] = normalized
            doc_name_key = normalized.get("doc_name")
            if doc_name_key and doc_name_key != normalized["doc_id"]:
                store[doc_name_key] = normalized

        return store

    def _collect_document_metadata(
        self,
        doc_id: Optional[str],
        result_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge metadata loaded from files with the fields that were stored in Qdrant payloads.
        """
        metadata: Dict[str, Any] = {}
        doc_name = result_metadata.get("doc_name")

        for key in filter(None, [doc_id, doc_name]):
            if key in self.metadata_store:
                metadata.update(self.metadata_store[key])

        for field in DOC_METADATA_FIELDS:
            value = result_metadata.get(field)
            if value not in (None, ""):
                metadata[field] = value

        if doc_id:
            metadata.setdefault("doc_id", doc_id)
        if doc_name:
            metadata.setdefault("doc_name", doc_name)

        return metadata

    def _build_filter(self, search_filter: Optional[SearchFilter]) -> Optional[Filter]:
        """
        Build Qdrant filter from SearchFilter.

        Args:
            search_filter: Search filter configuration

        Returns:
            Qdrant Filter object or None
        """
        if not search_filter or not self.config.enable_metadata_filter:
            return None

        conditions = []

        # Add specific field filters
        if search_filter.doc_id:
            conditions.append(
                FieldCondition(key="doc_id", match=MatchValue(value=search_filter.doc_id))
            )

        if search_filter.doc_name:
            conditions.append(
                FieldCondition(key="doc_name", match=MatchValue(value=search_filter.doc_name))
            )

        if search_filter.taxonomy:
            conditions.append(
                FieldCondition(key="taxonomy", match=MatchValue(value=search_filter.taxonomy))
            )

        if search_filter.origin:
            conditions.append(
                FieldCondition(key="origin", match=MatchValue(value=search_filter.origin))
            )

        if search_filter.owning_division:
            conditions.append(
                FieldCondition(key="owning_division", match=MatchValue(value=search_filter.owning_division))
            )

        if search_filter.section_level is not None:
            conditions.append(
                FieldCondition(key="section_level", match=MatchValue(value=search_filter.section_level))
            )

        if search_filter.has_section is not None:
            conditions.append(
                FieldCondition(key="has_section", match=MatchValue(value=search_filter.has_section))
            )

        # Full-text filter on text field
        if search_filter.text_contains:
            conditions.append(
                FieldCondition(key="text", match=MatchText(text=search_filter.text_contains))
            )

        # Add custom filters
        for key, value in search_filter.custom_filters.items():
            if isinstance(value, list):
                conditions.append(
                    FieldCondition(key=key, match=MatchAny(any=value))
                )
            elif isinstance(value, dict) and "min" in value or "max" in value:
                # Range filter
                range_filter = {}
                if "min" in value:
                    range_filter["gte"] = value["min"]
                if "max" in value:
                    range_filter["lte"] = value["max"]
                conditions.append(
                    FieldCondition(key=key, range=Range(**range_filter))
                )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        if not conditions:
            return None

        return Filter(must=conditions)


    async def _search_dense(
        self,
        query: str,
        limit: int,
        search_filter: Optional[Filter] = None
    ) -> List[SearchResult]:
        """
        Perform dense vector search.

        Args:
            query: Query text
            limit: Number of results
            search_filter: Optional filter

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = await self.dense_embedder.embed_query(query)

        # Perform search
        response = self.qdrant_client.query_points(
            collection_name=self.config.collection_name,
            query=query_embedding,
            using="dense",
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        results = response.points

        # Convert to SearchResult objects
        search_results = []
        for result in results:
            payload = result.payload or {}
            search_results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    doc_id=payload.get("doc_id", ""),
                    text=payload.get("text", "") if self.config.return_full_text else "",
                    score=result.score,
                    section_title=payload.get("section_title"),
                    section_breadcrumb=payload.get("section_breadcrumb", []),
                    metadata={
                        k: v for k, v in payload.items()
                        if k not in ["chunk_id", "doc_id", "text", "section_title", "section_breadcrumb"]
                    },
                    match_type="dense"
                )
            )

        return search_results

    async def _search_fts(
        self,
        query: str,
        limit: int,
        search_filter: Optional[Filter] = None
    ) -> List[SearchResult]:
        """
        Perform full-text search using Qdrant's text indexing.

        Args:
            query: Query text
            limit: Number of results
            search_filter: Optional filter

        Returns:
            List of search results
        """
        # Build filter conditions for FTS
        fts_conditions = []

        # Add text search on searchable_text field
        fts_conditions.append(
            FieldCondition(
                key="searchable_text",
                match=MatchText(text=query)
            )
        )

        # Combine with existing filter if provided
        if search_filter and search_filter.must:
            fts_conditions.extend(search_filter.must)

        fts_filter = Filter(must=fts_conditions)

        # Perform scroll search to get results based on text matching
        # Using scroll instead of search since we're not using vectors
        results, _ = self.qdrant_client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=fts_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        # Convert to SearchResult objects
        search_results = []
        for point in results:
            payload = point.payload or {}

            # Calculate a pseudo-score based on text matching
            # (Qdrant doesn't provide scores for scroll/filter operations)
            text_lower = query.lower()
            searchable_text = payload.get("searchable_text", "").lower()

            # Simple scoring: count occurrences of query terms
            query_terms = text_lower.split()
            score = sum(searchable_text.count(term) for term in query_terms) / len(query_terms)

            search_results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    doc_id=payload.get("doc_id", ""),
                    text=payload.get("text", "") if self.config.return_full_text else "",
                    score=score,
                    section_title=payload.get("section_title"),
                    section_breadcrumb=payload.get("section_breadcrumb", []),
                    metadata={
                        k: v for k, v in payload.items()
                        if k not in ["chunk_id", "doc_id", "text", "section_title", "section_breadcrumb", "searchable_text"]
                    },
                    match_type="fts"
                )
            )

        # Sort by score (since scroll doesn't sort)
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:limit]

    async def _search_hybrid_fusion(
        self,
        query: str,
        limit: int,
        search_filter: Optional[Filter] = None
    ) -> List[SearchResult]:
        """
        Perform advanced hybrid search with query API and fusion.

        This method:
        1. Retrieves candidates from both dense embeddings and FTS
        2. Applies fusion to combine results
        3. Reranks the final results based on configuration

        Args:
            query: Query text
            limit: Number of results
            search_filter: Optional filter

        Returns:
            List of search results
        """
        # For fusion mode, we'll combine dense search with FTS results
        # Since Qdrant's query API doesn't directly support FTS in prefetch,
        # we'll get both result sets and combine them manually

        # Generate dense embedding
        query_embedding = await self.dense_embedder.embed_query(query)

        # Get dense search results
        dense_results = await self._search_dense(query, self.config.prefetch_limit, search_filter)

        # Get FTS results
        fts_results = await self._search_fts(query, self.config.prefetch_limit, search_filter)

        # Combine and rerank results
        combined_results = self._merge_results(dense_results, fts_results, limit * 2)

        # Apply reranking if enabled
        if self.config.enable_reranking and self.config.rerank_with == "dense":
            # Rerank using dense embeddings
            # Get chunk IDs from combined results
            chunk_ids = [r.chunk_id for r in combined_results[:limit * 2]]

            if chunk_ids:
                # Perform dense search on the combined results to rerank
                reranked_response = self.qdrant_client.query_points(
                    collection_name=self.config.collection_name,
                    query=query_embedding,
                    using="dense",
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="chunk_id",
                                match=MatchAny(any=chunk_ids)
                            )
                        ]
                    ),
                    limit=limit,
                    with_payload=True
                )
                reranked = reranked_response.points

                # Convert reranked results
                search_results = []
                for result in reranked:
                    payload = result.payload or {}
                    search_results.append(
                        SearchResult(
                            chunk_id=payload.get("chunk_id", ""),
                            doc_id=payload.get("doc_id", ""),
                            text=payload.get("text", "") if self.config.return_full_text else "",
                            score=result.score,
                            section_title=payload.get("section_title"),
                            section_breadcrumb=payload.get("section_breadcrumb", []),
                            metadata={
                                k: v for k, v in payload.items()
                                if k not in ["chunk_id", "doc_id", "text", "section_title", "section_breadcrumb", "searchable_text"]
                            },
                            match_type="hybrid"
                        )
                    )
                return search_results

        # Return combined results without reranking
        return combined_results[:limit]

    async def search(
        self,
        query: str,
        search_filter: Optional[SearchFilter] = None,
        limit: Optional[int] = None,
        grouped: bool = True
    ) -> Union[List[SearchResult], Dict[str, Any]]:
        """
        Main search method supporting multiple search modes.

        Args:
            query: Search query
            search_filter: Optional metadata filters
            limit: Number of results (overrides config)
            grouped: If True (default) return results grouped by document metadata

        Returns:
            Either a grouped response dictionary (default) or a flat list of SearchResult objects
        """
        limit = limit or self.config.top_k

        # Build filter
        qdrant_filter = self._build_filter(search_filter)

        try:
            # Execute search based on mode
            if self.config.search_mode == SearchMode.DENSE:
                logger.info("Performing dense vector search")
                results = await self._search_dense(query, limit, qdrant_filter)

            elif self.config.search_mode == SearchMode.FTS:
                logger.info("Performing full-text search")
                results = await self._search_fts(query, limit, qdrant_filter)

            elif self.config.search_mode == SearchMode.HYBRID:
                logger.info("Performing hybrid search (FTS + Dense)")
                # Simple hybrid: run both searches and merge
                dense_results = await self._search_dense(
                    query,
                    int(limit * 1.5),  # Over-fetch for merging
                    qdrant_filter
                )
                fts_results = await self._search_fts(
                    query,
                    int(limit * 1.5),
                    qdrant_filter
                )

                # Merge results with weighted scores
                results = self._merge_results(
                    dense_results,
                    fts_results,
                    limit
                )

            elif self.config.search_mode == SearchMode.FUSION:
                logger.info("Performing advanced fusion search")
                results = await self._search_hybrid_fusion(query, limit, qdrant_filter)

            else:
                raise ValueError(f"Unknown search mode: {self.config.search_mode}")

            logger.info(f"Found {len(results)} results")

            if grouped:
                return self._build_grouped_response(query, results)

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def _merge_results(
        self,
        dense_results: List[SearchResult],
        fts_results: List[SearchResult],
        limit: int
    ) -> List[SearchResult]:
        """
        Merge dense and FTS results with weighted scoring.

        Args:
            dense_results: Results from dense search
            fts_results: Results from full-text search
            limit: Maximum results to return

        Returns:
            Merged and reranked results
        """
        # Create a dictionary to store merged results
        merged = {}

        # Add dense results with weighted scores
        for result in dense_results:
            key = result.chunk_id
            if key not in merged:
                result.score = result.score * self.config.dense_weight
                result.match_type = "dense"
                merged[key] = result
            else:
                # Combine scores if already present
                merged[key].score += result.score * self.config.dense_weight
                merged[key].match_type = "hybrid"

        # Add FTS results with weighted scores
        for result in fts_results:
            key = result.chunk_id
            if key not in merged:
                result.score = result.score * self.config.fts_weight
                result.match_type = "fts"
                merged[key] = result
            else:
                # Combine scores if already present
                merged[key].score += result.score * self.config.fts_weight
                if merged[key].match_type == "dense":
                    merged[key].match_type = "hybrid"

        # Sort by combined score and return top results
        sorted_results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:limit]

    def _build_grouped_response(
        self,
        query: str,
        results: List[SearchResult]
    ) -> Dict[str, Any]:
        """
        Group chunk-level results by document and attach shared metadata.

        Args:
            query: Original search query
            results: Flat list of chunk-level search results

        Returns:
            Dictionary summarizing matches grouped by document
        """
        documents: Dict[str, Dict[str, Any]] = {}

        for result in results:
            # Determine document identity and metadata
            doc_metadata = self._collect_document_metadata(result.doc_id, result.metadata)
            doc_key = doc_metadata.get("doc_id") or result.doc_id or doc_metadata.get("doc_name") or "unknown"

            if doc_key not in documents:
                documents[doc_key] = {
                    "doc_id": doc_metadata.get("doc_id") or result.doc_id or doc_key,
                    "doc_name": doc_metadata.get("doc_name"),
                    "metadata": doc_metadata,
                    "matches": [],
                    "best_score": result.score
                }

            document_entry = documents[doc_key]
            document_entry["best_score"] = max(document_entry["best_score"], result.score)

            # Chunk-specific metadata (exclude document-level fields)
            chunk_metadata = {
                k: v for k, v in result.metadata.items()
                if k not in DOC_METADATA_FIELDS
            }

            document_entry["matches"].append({
                "id": result.chunk_id,
                "chunk_id": result.chunk_id,
                "score": result.score,
                "match_type": result.match_type,
                "text": result.text,
                "section_title": result.section_title,
                "section_breadcrumb": result.section_breadcrumb,
                "metadata": chunk_metadata
            })

        # Convert to sorted list (highest scoring doc first)
        sorted_documents = sorted(
            documents.values(),
            key=lambda entry: entry["best_score"],
            reverse=True
        )

        # Remove helper key
        for entry in sorted_documents:
            entry.pop("best_score", None)

        return {
            "query": query,
            "total_matches": len(results),
            "document_count": len(sorted_documents),
            "documents": sorted_documents
        }

    async def get_similar_chunks(
        self,
        chunk_id: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """
        Find similar chunks to a given chunk ID.

        Args:
            chunk_id: ID of the reference chunk
            limit: Number of similar chunks to return

        Returns:
            List of similar chunks
        """
        # Retrieve the reference chunk
        reference_chunks = self.qdrant_client.retrieve(
            collection_name=self.config.collection_name,
            ids=[int(chunk_id) if chunk_id.isdigit() else hash(chunk_id) % (10 ** 8)],
            with_vectors=True,
            with_payload=True
        )

        if not reference_chunks:
            logger.warning(f"Chunk {chunk_id} not found")
            return []

        reference = reference_chunks[0]

        # Use the dense vector for similarity search
        if "dense" in reference.vector:
            response = self.qdrant_client.query_points(
                collection_name=self.config.collection_name,
                query=reference.vector["dense"],
                using="dense",
                limit=limit + 1,  # +1 to exclude self
                with_payload=True
            )
            results = response.points

            # Convert and filter out the reference chunk
            search_results = []
            for result in results:
                if result.id != reference.id:
                    payload = result.payload or {}
                    search_results.append(
                        SearchResult(
                            chunk_id=payload.get("chunk_id", ""),
                            doc_id=payload.get("doc_id", ""),
                            text=payload.get("text", "") if self.config.return_full_text else "",
                            score=result.score,
                            section_title=payload.get("section_title"),
                            section_breadcrumb=payload.get("section_breadcrumb", []),
                            metadata={
                                k: v for k, v in payload.items()
                                if k not in ["chunk_id", "doc_id", "text", "section_title", "section_breadcrumb"]
                            },
                            match_type="similarity"
                        )
                    )

            return search_results[:limit]

        return []

    def get_search_field_info(self) -> Dict[str, Any]:
        """
        Get information about what fields are being searched.

        Returns:
            Dictionary with search field information
        """
        return {
            "search_includes": [
                "section_breadcrumb",  # Full section hierarchy
                "section_title",       # Individual section title
                "text"                 # Main chunk text
            ],
            "search_methods": {
                "dense": "Semantic search using OpenAI embeddings",
                "fts": "Full-text search using Qdrant's text indexing",
                "hybrid": "Combined FTS + semantic search"
            },
            "description": (
                "Search works on a combined representation of:\n"
                "1. Section breadcrumb (full hierarchy path)\n"
                "2. Section title (if available)\n"
                "3. Main text content\n\n"
                "- Dense search: Uses OpenAI embeddings for semantic understanding\n"
                "- FTS: Uses Qdrant's built-in text indexing for keyword matching\n"
                "- Hybrid: Combines both approaches with weighted scoring\n"
                "- Fusion: Advanced mode with result merging and reranking"
            ),
            "example": {
                "chunk": {
                    "section_breadcrumb": ["UBS Reports", "Financial Performance", "Q3 Results"],
                    "section_title": "Q3 Results",
                    "text": "Revenue increased by 15% year-over-year..."
                },
                "searchable_text": (
                    "Section: UBS Reports > Financial Performance > Q3 Results\n"
                    "Title: Q3 Results\n"
                    "Revenue increased by 15% year-over-year..."
                )
            }
        }

    async def close(self):
        """Clean up resources"""
        await self.dense_embedder.close()


async def main():
    """
    Main function to demonstrate search capabilities.
    """
    # Configure search
    config = SearchConfig(
        search_mode=SearchMode.HYBRID,
        top_k=5,
        return_full_text=True
    )

    # Create searcher
    searcher = HybridSearcher(config)

    try:
        # Example 1: Simple search
        print("\n=== Simple Search ===")
        results = await searcher.search(
            query="financial performance and results",
            limit=3
        )

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Chunk ID: {result.chunk_id}")
            print(f"   Score: {result.score:.4f} (Type: {result.match_type})")
            print(f"   Section: {result.section_title or 'N/A'}")
            print(f"   Text Preview: {result.text[:200]}...")

        # Example 2: Full-text search
        print("\n=== Full-Text Search ===")
        searcher.config.search_mode = SearchMode.FTS
        results = await searcher.search(
            query="integration progress Swiss economy",
            limit=3
        )

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Chunk ID: {result.chunk_id}")
            print(f"   Score: {result.score:.4f} (Type: {result.match_type})")
            print(f"   Section: {result.section_title or 'N/A'}")
            print(f"   Text Preview: {result.text[:200]}...")

        # Example 3: Filtered search
        print("\n=== Filtered Search ===")
        searcher.config.search_mode = SearchMode.HYBRID
        search_filter = SearchFilter(
            doc_name="ubs_report",
            has_section=True,
            text_contains="integration"
        )

        results = await searcher.search(
            query="progress on integration",
            search_filter=search_filter,
            limit=3
        )

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Chunk ID: {result.chunk_id}")
            print(f"   Score: {result.score:.4f}")
            print(f"   Section: {result.section_title or 'N/A'}")
            print(f"   Metadata: {json.dumps(result.metadata, indent=2)}")

    finally:
        await searcher.close()


if __name__ == "__main__":
    asyncio.run(main())

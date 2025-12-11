"""
Complete Example: Qdrant Hybrid Search Implementation

This example demonstrates the complete workflow of:
1. Setting up the environment
2. Ingesting documents with metadata
3. Performing various types of searches
4. Using metadata filters
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from provider.qdrant import (
    DocumentIngester,
    IngestionConfig,
    HybridSearcher,
    SearchConfig,
    SearchFilter,
    SearchMode
)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def print_results(response: Dict[str, Any], verbose: bool = False):
    """Pretty print grouped search results"""
    documents = response.get("documents", [])
    if not documents:
        print("\nNo results found.")
        return

    print(f"\nQuery: {response.get('query')}")
    print(f"Documents matched: {response.get('document_count', 0)} | Total chunks: {response.get('total_matches', 0)}")

    for i, doc in enumerate(documents, 1):
        doc_title = (
            doc.get("metadata", {}).get("title")
            or doc.get("doc_name")
            or doc.get("doc_id")
            or f"Document {i}"
        )
        print(f"\n{i}. Document: {doc_title}")
        print(f"   Doc ID: {doc.get('doc_id')}")

        if doc.get("metadata"):
            print("   Metadata:")
            for key, value in doc["metadata"].items():
                print(f"     - {key}: {value}")

        for match in doc.get("matches", []):
            print(f"   • Chunk: {match.get('id')}")
            print(f"     Score: {match.get('score', 0):.4f} | Type: {match.get('match_type')}")

            if match.get("section_title"):
                print(f"     Section: {match['section_title']}")

            breadcrumb = match.get("section_breadcrumb") or []
            if breadcrumb:
                print(f"     Path: {' > '.join(breadcrumb)}")

            text = match.get("text") or "N/A"
            text_preview = text[:300]
            if "..." not in text_preview and len(text) > 300:
                text_preview += "..."
            print(f"     Text: {text_preview}")

            if verbose and match.get("metadata"):
                print("     Chunk metadata:")
                for key, value in match["metadata"].items():
                    print(f"       - {key}: {value}")


async def run_ingestion_example():
    """Demonstrate document ingestion"""
    print_section("DOCUMENT INGESTION")

    # Configure ingestion
    config = IngestionConfig(
        chunks_file="/Users/preetam/Develop/mcp_servers/doc-mcp/parser/chunks.json",
        metadata_file="/Users/preetam/Develop/mcp_servers/doc-mcp/parser/chunks_meta_data.json",
        collection_name="documents",
        recreate_collection=True,  # Start fresh
        batch_size=50,
        enable_text_index=True  # Enable full-text search for hybrid mode
    )

    # Create ingester
    ingester = DocumentIngester(config)

    # Run ingestion
    print("\nStarting document ingestion...")
    results = await ingester.ingest_documents()

    print(f"\nIngestion completed!")
    print(f"  Status: {results['status']}")
    print(f"  Total chunks processed: {results['total_chunks']}")
    print(f"  Points indexed: {results['total_points_indexed']}")
    print(f"  Collection: {results['collection_name']}")
    print(f"  Vector types: {', '.join(results['indexed_vectors'])}")
    print(f"  Text indexing enabled: {results['text_index_enabled']}")

    # Verify ingestion
    print("\nVerifying ingestion...")
    verification = await ingester.verify_ingestion()
    print(f"  Points in collection: {verification['points_count']}")
    print(f"  Vectors configured: {len(verification['indexed_vectors_config'])}")


async def run_search_examples():
    """Demonstrate various search capabilities"""
    print_section("SEARCH EXAMPLES")

    # Initialize searcher with hybrid mode and reranking
    config = SearchConfig(
        collection_name="documents",
        search_mode=SearchMode.HYBRID,
        top_k=5,
        return_full_text=True,
        fts_weight=0.3,
        dense_weight=0.7,
        enable_reranking=True,
        rerank_with="dense"
    )

    searcher = HybridSearcher(config)

    # Show what fields are being searched
    print("\n📍 SEARCH FIELD INFORMATION:")
    field_info = searcher.get_search_field_info()
    print(field_info['description'])
    print(f"\nSearchable fields: {', '.join(field_info['search_includes'])}")

    try:
        # Example 1: Simple semantic search
        print("\n1. SEMANTIC SEARCH (Dense Vectors Only)")
        print("   Query: 'financial performance and quarterly results'")

        searcher.config.search_mode = SearchMode.DENSE
        results = await searcher.search(
            query="financial performance and quarterly results",
            limit=3
        )
        print_results(results)

        # Example 2: Full-text search
        print("\n2. FULL-TEXT SEARCH (FTS Only)")
        print("   Query: 'USD billion integration'")

        searcher.config.search_mode = SearchMode.FTS
        results = await searcher.search(
            query="USD billion integration",
            limit=3
        )
        print_results(results)

        # Example 3: Hybrid search
        print("\n3. HYBRID SEARCH (FTS + Dense)")
        print("   Query: 'client momentum and asset growth'")

        searcher.config.search_mode = SearchMode.HYBRID
        results = await searcher.search(
            query="client momentum and asset growth",
            limit=3
        )
        print_results(results)

        # Example 4: Filtered search
        print("\n4. FILTERED SEARCH")
        print("   Query: 'integration progress'")
        print("   Filters: has_section=True, text_contains='Swiss'")

        search_filter = SearchFilter(
            has_section=True,
            text_contains="Swiss"
        )

        results = await searcher.search(
            query="integration progress",
            search_filter=search_filter,
            limit=3
        )
        print_results(results, verbose=True)

        # Example 5: Section-level search
        print("\n5. SECTION-LEVEL SEARCH")
        print("   Query: 'financial performance'")
        print("   Filter: section_level=3 (subsections only)")

        search_filter = SearchFilter(
            section_level=3
        )

        results = await searcher.search(
            query="financial performance",
            search_filter=search_filter,
            limit=3
        )
        print_results(results)

        # Example 6: Advanced metadata filtering
        print("\n6. ADVANCED METADATA FILTERING")
        print("   Query: 'revenue growth'")
        print("   Custom filters: text_length range")

        search_filter = SearchFilter(
            custom_filters={
                "text_length": {"min": 500, "max": 2000}
            }
        )

        results = await searcher.search(
            query="revenue growth",
            search_filter=search_filter,
            limit=3
        )

        print_results(results)
        # Show text lengths for each match
        for doc in results.get("documents", []):
            for i, match in enumerate(doc.get("matches", []), 1):
                text_len = match.get("metadata", {}).get('text_length', 0)
                print(f"   Match {i} text length: {text_len} characters")

        # Example 7: Searching by section titles and breadcrumbs
        print("\n7. SECTION TITLE/BREADCRUMB SEARCH")
        print("   Query: 'Group summary Strong financial performance'")
        print("   (This searches across section breadcrumb, title, AND text)")

        results = await searcher.search(
            query="Group summary Strong financial performance",
            limit=3
        )

        documents = results.get("documents", [])
        if documents:
            for doc in documents:
                print(f"\nDocument: {doc.get('doc_id')}")
                for match in doc.get("matches", []):
                    print(f"   - Chunk: {match.get('id')} | Score: {match.get('score', 0):.4f}")
                    breadcrumb = match.get("section_breadcrumb") or []
                    if breadcrumb:
                        print(f"     📍 Breadcrumb: {' > '.join(breadcrumb)}")
                    section = match.get("section_title")
                    if section:
                        print(f"     📍 Section: {section}")
                    text = match.get("text") or "N/A"
                    preview = text[:200]
                    if "..." not in preview and len(text) > 200:
                        preview += "..."
                    print(f"     Text: {preview}")

        # Example 8: Demonstrating Fusion with Reranking
        print("\n8. FUSION MODE WITH RERANKING")
        print("   Query: 'client momentum asset growth'")
        print("   Mode: FUSION with dense reranking")

        searcher.config.search_mode = SearchMode.FUSION
        searcher.config.enable_reranking = True
        searcher.config.rerank_with = "dense"
        searcher.config.prefetch_limit = 50  # Get more candidates for reranking

        results = await searcher.search(
            query="client momentum asset growth",
            limit=3
        )

        print("\n   ✅ Results after fusion and reranking:")
        print_results(results)

    finally:
        await searcher.close()


async def run_similarity_search():
    """Demonstrate finding similar chunks"""
    print_section("SIMILARITY SEARCH")

    config = SearchConfig(
        collection_name="documents",
        return_full_text=True
    )

    searcher = HybridSearcher(config)

    try:
        # First, find a chunk about a specific topic
        print("\nFinding reference chunk about 'integration'...")
        flat_results = await searcher.search(
            query="integration progress",
            limit=1,
            grouped=False
        )

        if flat_results:
            reference_chunk = flat_results[0]
            print(f"\nReference chunk: {reference_chunk.chunk_id}")
            print(f"Text preview: {reference_chunk.text[:200]}...")

            # Find similar chunks
            print(f"\nFinding chunks similar to {reference_chunk.chunk_id}...")
            similar = await searcher.get_similar_chunks(
                chunk_id=reference_chunk.chunk_id,
                limit=3
            )

            print("\nSimilar chunks:")
            if similar:
                for i, result in enumerate(similar, 1):
                    print(f"\n{i}. Chunk: {result.chunk_id}")
                    print(f"   Score: {result.score:.4f} | Doc: {result.doc_id}")
                    text = result.text or "N/A"
                    preview = text[:200]
                    if "..." not in preview and len(text) > 200:
                        preview += "..."
                    print(f"   Text: {preview}")
            else:
                print("No similar chunks found.")

    finally:
        await searcher.close()


async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print(" QDRANT HYBRID SEARCH - COMPLETE EXAMPLE")
    print("=" * 60)

    # Check environment variables
    print("\nChecking environment configuration...")

    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    ]

    print("✓ Environment configured")

    # Check if Qdrant is running
    print("\nChecking Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print(f"✓ Connected to Qdrant (found {len(collections.collections)} collections)")
    except Exception as e:
        print(f"\n⚠️  Could not connect to Qdrant: {e}")
        print("\nPlease ensure Qdrant is running on localhost:6333")
        print("You can start it with: docker run -p 6333:6333 qdrant/qdrant")
        return

    # Check if data files exist
    print("\nChecking data files...")
    chunks_path = Path("/Users/preetam/Develop/mcp_servers/doc-mcp/parser/chunks.json")
    metadata_path = Path("/Users/preetam/Develop/mcp_servers/doc-mcp/parser/chunks_meta_data.json")

    if not chunks_path.exists():
        print(f"⚠️  Chunks file not found: {chunks_path}")
        return
    if not metadata_path.exists():
        print(f"⚠️  Metadata file not found: {metadata_path}")
        return

    print("✓ Data files found")

    # Run examples
    try:
        # Step 1: Ingest documents
        await run_ingestion_example()

        # Step 2: Demonstrate search capabilities
        await run_search_examples()

        # Step 3: Demonstrate similarity search
        await run_similarity_search()

        print_section("EXAMPLES COMPLETED")
        print("\nAll examples completed successfully!")
        print("\nYou can now use the HybridSearcher in your application:")
        print("  from provider.qdrant import HybridSearcher, SearchConfig, SearchFilter")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Load environment variables if .env file exists
    from pathlib import Path
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()

    # Run the examples
    asyncio.run(main())

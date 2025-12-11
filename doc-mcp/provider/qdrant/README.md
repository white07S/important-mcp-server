# Qdrant Hybrid Search Implementation

A powerful document search system that combines semantic search (dense embeddings) with full-text search (FTS) using Qdrant vector database and Azure OpenAI embeddings.

## Features

- **Multi-Field Search**: Searches across section breadcrumbs, titles, AND text content simultaneously
- **Advanced Reranking**: Configurable reranking strategies for optimal result ordering
- **Hybrid Search**: Combines semantic understanding with keyword precision using FTS
- **Multiple Search Modes**:
  - Dense vector search (semantic understanding via OpenAI embeddings)
  - Full-text search (keyword matching using Qdrant's text indexing)
  - Hybrid search (combined FTS + dense with weighted scoring)
  - Fusion search (merge + rerank for optimal results)
- **Metadata Filtering**: Filter results by document properties
- **Azure OpenAI Integration**: Uses text-embedding-3-large for high-quality embeddings
- **Async Support**: Fully asynchronous for better performance
- **Batch Processing**: Efficient ingestion of large document sets

## Architecture

```
provider/
├── embeddings/
│   ├── __init__.py
│   └── azure_openai.py      # Azure OpenAI embeddings provider
└── qdrant/
    ├── __init__.py
    ├── ingest.py             # Document ingestion module
    ├── search.py             # Hybrid search implementation
    ├── example.py            # Complete usage examples
    └── README.md             # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `qdrant-client>=1.7.0`
- `openai>=1.0.0`
- `fastembed>=0.2.0`
- `numpy>=1.24.0`
- `tqdm>=4.65.0`
- `python-dotenv>=1.0.0` (optional)

### 2. Start Qdrant

Run Qdrant using Docker:

```bash
docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

Or using Docker Compose:

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

### 3. Configure Azure OpenAI

Create a `.env` file in your project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

## Usage

### Document Ingestion

```python
from provider.qdrant import DocumentIngester, IngestionConfig

# Configure ingestion
config = IngestionConfig(
    chunks_file="parser/chunks.json",
    metadata_file="parser/chunks_meta_data.json",
    collection_name="documents",
    recreate_collection=True,
    batch_size=50,
    enable_text_index=True  # Enable full-text search for hybrid mode
)

# Create ingester and run
ingester = DocumentIngester(config)
results = await ingester.ingest_documents()
```

### Hybrid Search

```python
from provider.qdrant import HybridSearcher, SearchConfig, SearchFilter, SearchMode

# Configure search
config = SearchConfig(
    collection_name="documents",
    search_mode=SearchMode.HYBRID,
    top_k=10,
    fts_weight=0.3,
    dense_weight=0.7,
    metadata_paths=["/Users/preetam/Develop/mcp_servers/doc-mcp/parser/chunks_meta_data.json"]  # optional override
)

# Create searcher
searcher = HybridSearcher(config)

# Simple search (grouped by document)
response = await searcher.search(
    query="financial performance and growth",
    limit=5
)

# Filtered search with metadata
search_filter = SearchFilter(
    doc_name="ubs_report",
    has_section=True,
    text_contains="integration"
)

response = await searcher.search(
    query="progress on integration",
    search_filter=search_filter,
    limit=5
)

# Access grouped results
for document in response["documents"]:
    title = document["metadata"].get("title") or document["doc_id"]
    print(f"\nDocument: {title}")
    for match in document["matches"]:
        print(f"  - Chunk {match['id']} | Score: {match['score']:.4f}")
        print(f"    Text: {match['text'][:200]}...")

# Need raw chunk results instead? Pass grouped=False
flat_results = await searcher.search(
    query="financial performance and growth",
    limit=5,
    grouped=False
)

```

Document-level metadata bundled with each result is loaded automatically from `parser/chunks_meta_data.json` (if it exists) or any custom files supplied through `metadata_paths`. This ensures every group of chunk matches carries a single, authoritative metadata dictionary sourced from your JSON configuration.

### How Multi-Field Search Works

The system creates a combined searchable text from each chunk that includes:

1. **Section Breadcrumb**: The full hierarchy path (e.g., "UBS Reports > Financial Performance > Q3 Results")
2. **Section Title**: The specific section name
3. **Text Content**: The main chunk text

This combined representation is used to generate dense embeddings and create text indexes, ensuring that searches can match on:
- Document structure and organization (via breadcrumbs)
- Section headings and titles
- Actual content text

Example of what gets indexed:
```
Section: UBS Reports > Financial Performance > Q3 Results
Title: Q3 Results
Revenue increased by 15% year-over-year...
```

### Reranking Strategy

The system supports advanced reranking to improve result quality:

1. **Search Phase**: Retrieves candidates from both dense embeddings and full-text search
2. **Fusion Phase**: Combines results using configurable fusion methods (RRF or DBSF)
3. **Rerank Phase**: Re-scores the combined results using:
   - **Dense reranking** (default): Uses semantic embeddings for final scoring
   - **FTS reranking**: Uses keyword relevance for final scoring
   - **Cross reranking**: Combines both approaches

Configure reranking:
```python
config = SearchConfig(
    enable_reranking=True,
    rerank_with="dense",  # or "fts" or "cross"
    prefetch_limit=100,   # Number of candidates to consider
    fts_weight=0.3,       # Weight for FTS results
    dense_weight=0.7      # Weight for dense results
)
```

### Search Modes

#### 1. Dense Search (Semantic)
Best for understanding context and meaning:

```python
config.search_mode = SearchMode.DENSE
response = await searcher.search("What are the financial implications?")
```

#### 2. Full-Text Search (Keyword)
Best for specific terms and exact matches:

```python
config.search_mode = SearchMode.FTS
response = await searcher.search("USD 2.5 billion Q3 2025")
```

#### 3. Hybrid Search
Combines both approaches:

```python
config.search_mode = SearchMode.HYBRID
response = await searcher.search("revenue growth in Asian markets")
```

#### 4. Fusion Search
Advanced mode with result merging and reranking:

```python
config.search_mode = SearchMode.FUSION
response = await searcher.search("strategic initiatives and market expansion")
```

### Metadata Filtering

The system supports comprehensive metadata filtering:

```python
# Filter by document properties
filter = SearchFilter(
    doc_id="ubs_report",
    doc_name="UBS Annual Report",
    taxonomy="financial_reports",
    origin="UBS AG",
    owning_division="Corporate Communications"
)

# Filter by section properties
filter = SearchFilter(
    section_level=3,  # Only subsections
    has_section=True,  # Only chunks with section titles
    text_contains="integration"  # Text must contain this word
)

# Custom range filters
filter = SearchFilter(
    custom_filters={
        "text_length": {"min": 500, "max": 2000},
        "chunk_index": {"max": 50}
    }
)

# Combine multiple filters
filter = SearchFilter(
    doc_name="ubs_report",
    has_section=True,
    section_level=2,
    text_contains="financial",
    custom_filters={
        "text_length": {"min": 100}
    }
)
```

### Finding Similar Chunks

```python
# Find chunks similar to a specific chunk
similar = await searcher.get_similar_chunks(
    chunk_id="ubs_report::s1::c0",
    limit=5
)
```

## Data Format

### chunks.json
Array of document chunks:

```json
[
  {
    "id": "doc_id::s1::c0",
    "doc_id": "doc_id",
    "text": "Actual text content...",
    "section_title": "Section Title",
    "section_level": 1,
    "section_breadcrumb": ["Main", "Sub", "Title"]
  }
]
```

### chunks_meta_data.json
Document-level metadata:

```json
{
  "doc_name": "document_name",
  "file_path": "path/to/original.pdf",
  "parsed_file_path": "path/to/chunks.json",
  "taxonomy": "category",
  "last_modified": "2024-06-10T12:00:00Z",
  "title": "Document Title",
  "summary": "Brief summary...",
  "origin": "Source Organization",
  "owning_division": "Department"
}
```

## Running Examples

Run the complete example to see all features:

```bash
cd /path/to/doc-mcp
python provider/qdrant/example.py
```

The example demonstrates:
1. Document ingestion with metadata
2. Various search modes
3. Metadata filtering
4. Similarity search
5. Result ranking and scoring

## Performance Tips

1. **Batch Size**: Adjust `batch_size` in `IngestionConfig` based on your system resources
2. **Search Weights**: Tune `fts_weight` and `dense_weight` for your use case
3. **Prefetch Limit**: Increase `prefetch_limit` for better recall at the cost of speed
4. **Text Indexing**: Configure tokenizer type (word, multilingual) based on your content

## Troubleshooting

### Connection Issues
- Ensure Qdrant is running on `localhost:6333`
- Check firewall settings
- Verify Docker is running

### Embedding Errors
- Verify Azure OpenAI credentials in `.env`
- Check API quota and limits
- Ensure deployment name matches

### Search Quality
- Adjust weight parameters
- Try different search modes
- Increase `top_k` for more results
- Use metadata filters to narrow results

## Advanced Configuration

### Custom Embedding Dimensions

```python
from provider.embeddings import EmbeddingConfig

embedding_config = EmbeddingConfig(
    dimensions=1536,  # For text-embedding-ada-002
    batch_size=100
)
```

### Text Indexing Configuration

```python
config = IngestionConfig(
    enable_text_index=True,
    text_index_tokenizer="word"  # or "multilingual" for non-English content
)
```

### Collection Configuration

```python
from qdrant_client.models import Distance

# Custom distance metrics
vectors_config = {
    "dense": VectorParams(
        size=3072,
        distance=Distance.DOT  # Or EUCLID, MANHATTAN
    )
}
```

## API Reference

### DocumentIngester

- `ingest_documents()`: Ingest documents from JSON files
- `verify_ingestion()`: Verify successful ingestion

### HybridSearcher

- `search(query, search_filter, limit)`: Main search method
- `get_similar_chunks(chunk_id, limit)`: Find similar chunks

### SearchFilter

- `doc_id`: Filter by document ID
- `doc_name`: Filter by document name
- `taxonomy`: Filter by taxonomy
- `section_level`: Filter by section depth
- `has_section`: Filter chunks with/without sections
- `text_contains`: Full-text filter
- `custom_filters`: Custom field filters

### SearchResult

- `chunk_id`: Unique chunk identifier
- `doc_id`: Parent document ID
- `text`: Chunk text content
- `score`: Relevance score
- `section_title`: Section title if available
- `section_breadcrumb`: Section hierarchy
- `metadata`: Additional metadata
- `match_type`: Type of match (dense/fts/hybrid)

## License

This implementation is designed for document search and retrieval in production environments.

## Support

For issues or questions, please refer to the Qdrant documentation:
- [Qdrant Python Client](https://python-client.qdrant.tech/)
- [Qdrant Hybrid Search](https://qdrant.tech/articles/hybrid-search/)
- [Azure OpenAI Embeddings](https://learn.microsoft.com/en-us/azure/ai-services/openai/)

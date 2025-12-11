## Doc MCP

This project exposes a single FastMCP tool that performs semantic search over the
pre-chunked UBS report stored in Qdrant. The searcher combines Azure OpenAI
embeddings with Qdrant's dense and full-text capabilities and returns structured,
metadata-enriched chunks grouped by document.

### Prerequisites

1. A running Qdrant instance (defaults assume `localhost:6333`).
2. Documents ingested into Qdrant by running the helper in `provider/qdrant/ingest.py`.
3. Azure OpenAI credentials available as environment variables (see
   `provider/embeddings/azure_openai.py`).

### Running the MCP server

```bash
uv run fastmcp run doc_mcp/server.py
```

Or directly:

```bash
uv run python doc_mcp/server.py
```

Environment overrides:

| Variable | Default | Description |
| --- | --- | --- |
| `DOC_MCP_QDRANT_HOST` | `localhost` | Qdrant host |
| `DOC_MCP_QDRANT_PORT` | `6333` | Qdrant port |
| `DOC_MCP_COLLECTION` | `documents` | Collection name |
| `DOC_MCP_DEFAULT_TOPK` | `5` | Default top_k value |
| `DOC_MCP_METADATA_PATHS` | (auto-detect) | OS-path-separated list of metadata JSON files |

### Example MCP configuration

```toml
[mcp_servers.doc-search]
command = "/opt/homebrew/bin/uv"
args = [
  "--project", "/Users/preetam/Develop/mcp_servers/doc-mcp",
  "run", "fastmcp", "run", "doc_mcp/server.py",
]
```

Once configured, the single `search_documents` tool is available for Codex or
any MCP-compatible client. Provide a natural language `query` and optionally
override `top_k` (1–20). The response includes grouped matches along with their
document-level metadata pulled from `parser/chunks_meta_data.json`.

"""FastMCP server exposing Qdrant-backed document search as a single tool."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

# Ensure project root (which contains provider/) is importable when this module is run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from provider.qdrant import HybridSearcher, SearchConfig, SearchMode

# Holder for the long-lived HybridSearcher managed by the FastMCP lifespan hook.
_SEARCHER: HybridSearcher | None = None


def _metadata_paths_from_env() -> list[str] | None:
    """Parse DOC_MCP_METADATA_PATHS into a list of filesystem paths."""
    raw_value = os.getenv("DOC_MCP_METADATA_PATHS")
    if not raw_value:
        return None

    paths = [part.strip() for part in raw_value.split(os.pathsep) if part.strip()]
    return paths or None


def _build_searcher() -> HybridSearcher:
    """Instantiate the Qdrant HybridSearcher using environment overrides when available."""

    metadata_paths = _metadata_paths_from_env()
    config = SearchConfig(
        qdrant_host=os.getenv("DOC_MCP_QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("DOC_MCP_QDRANT_PORT", "6333")),
        collection_name=os.getenv("DOC_MCP_COLLECTION", "documents"),
        top_k=int(os.getenv("DOC_MCP_DEFAULT_TOPK", "5")),
        metadata_paths=metadata_paths,
        search_mode=SearchMode.HYBRID,
        return_full_text=True,
    )
    return HybridSearcher(config)


def _get_searcher() -> HybridSearcher:
    if _SEARCHER is None:
        raise RuntimeError("Search service is not ready yet. Try again in a moment.")
    return _SEARCHER


@asynccontextmanager
async def lifespan(_server: FastMCP):  # type: ignore[name-defined]
    """Create and clean up the HybridSearcher for the server lifetime."""

    global _SEARCHER
    _SEARCHER = _build_searcher()
    try:
        yield {"searcher": "qdrant"}
    finally:
        if _SEARCHER is not None:
            await _SEARCHER.close()
            _SEARCHER = None


mcp = FastMCP(
    "doc_mcp",
    instructions=(
        "Provides semantic search over pre-chunked UBS documents backed by Qdrant. "
        "Use the single tool to retrieve the most relevant context snippets grouped by document."
    ),
    lifespan=lifespan,
)


def _format_grouped_results(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize HybridSearcher grouped output into a Codex-friendly structure."""

    documents: list[dict[str, Any]] = []
    for doc in raw.get("documents", []):
        matches: list[dict[str, Any]] = []
        for match in doc.get("matches", []):
            chunk_metadata = match.get("metadata") or {}
            matches.append(
                {
                    "id": match.get("id"),
                    "doc_id": doc.get("doc_id"),
                    "text": match.get("text"),
                    "score": match.get("score"),
                    "match_type": match.get("match_type"),
                    "section_title": match.get("section_title"),
                    "section_breadcrumb": match.get("section_breadcrumb"),
                    "section_level": chunk_metadata.get("section_level"),
                    "metadata": chunk_metadata,
                }
            )

        documents.append(
            {
                "doc_id": doc.get("doc_id"),
                "doc_name": doc.get("doc_name"),
                "metadata": doc.get("metadata") or {},
                "matches": matches,
            }
        )

    return {
        "query": raw.get("query"),
        "total_matches": raw.get("total_matches"),
        "document_count": raw.get("document_count"),
        "documents": documents,
    }


@mcp.resource("doc-mcp://tools")
def describe_tools() -> str:
    """Advertise available tools so MCP clients can introspect before invoking."""

    tools = [
        {
            "name": "search_documents",
            "description": "Hybrid semantic search over the UBS report stored in Qdrant.",
            "inputs": {
                "query": "Natural language question or keywords to search for.",
                "top_k": "Optional integer between 1 and 20. Defaults to 5.",
            },
            "example": {
                "query": "client momentum and asset growth",
                "top_k": 5,
            },
            "output": "JSON object containing grouped matches with document-level metadata and snippet text.",
        }
    ]

    payload = {
        "tool_count": len(tools),
        "tools": tools,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


@mcp.tool(name="search_documents", description="Search pre-indexed UBS documents via Qdrant")
async def search_documents(
    query: Annotated[str, Field(description="Natural language query to search across the indexed documents.")],
    top_k: Annotated[
        int,
        Field(
            default=5,
            ge=1,
            le=10,
            description="Maximum number of top matching chunks to retrieve before grouping by document.",
        ),
    ] = 5,
) -> dict[str, Any]:
    """Return the best-matching document chunks grouped by document metadata."""

    query_text = query.strip()
    if not query_text:
        raise ValueError("Query cannot be empty.")

    limit = min(top_k, 10)
    searcher = _get_searcher()
    raw_results = await searcher.search(query=query_text, limit=limit, grouped=True)
    formatted = _format_grouped_results(raw_results)
    formatted["requested_top_k"] = top_k
    return formatted


if __name__ == "__main__":
    mcp.run()

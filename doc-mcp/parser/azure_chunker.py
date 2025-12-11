#!/usr/bin/env python3
"""
Advanced Azure Document Intelligence Chunker with Structure Preservation

Features:
- Hierarchical section tracking with breadcrumbs
- Atomic element handling (tables, figures, lists stay intact)
- Configurable overlap between chunks
- Semantic boundary detection (avoid mid-sentence splits)
- Rich metadata for each chunk (pages, sections, element types)
- JSON output ready for vector DB ingestion

Usage:
    python azure_di_chunker.py input.pdf --api-key YOUR_KEY --endpoint YOUR_ENDPOINT
    python azure_di_chunker.py input.pdf --api-key YOUR_KEY --endpoint YOUR_ENDPOINT --output chunks.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class ElementKind(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    SECTION = "section"
    LIST = "list"
    CODE = "code"
    FORMULA = "formula"


HEADING_ROLES = {"title", "sectionHeading", "pageHeader", "pageFooter"}
ATOMIC_KINDS = {ElementKind.TABLE, ElementKind.FIGURE, ElementKind.CODE, ElementKind.FORMULA}


@dataclass
class NormalizedElement:
    """Normalized representation of a DI element with unified span info."""
    kind: ElementKind
    span_start: int
    span_end: int
    role: Optional[str] = None
    heading_level: Optional[int] = None
    pages: list[int] = field(default_factory=list)
    is_atomic: bool = False  # Should not be split
    is_list_item: bool = False
    list_group_id: Optional[str] = None
    raw: Any = None

    @property
    def length(self) -> int:
        return self.span_end - self.span_start


@dataclass
class SectionNode:
    """Hierarchical section node for breadcrumb tracking."""
    title: Optional[str]
    level: int
    element_indices: list[int] = field(default_factory=list)
    parent: Optional[SectionNode] = None
    
    def get_breadcrumb(self) -> list[str]:
        """Build breadcrumb path from root to this section."""
        path = []
        node = self
        while node is not None:
            if node.title:
                path.append(node.title)
            node = node.parent
        return list(reversed(path))
    
    def get_breadcrumb_str(self, separator: str = " > ") -> str:
        return separator.join(self.get_breadcrumb())


@dataclass
class Chunk:
    """Final chunk ready for vector DB ingestion."""
    id: str
    doc_id: str
    text: str
    
    # Section metadata
    section_title: Optional[str] = None
    section_level: Optional[int] = None
    section_breadcrumb: list[str] = field(default_factory=list)
    
    # Span info for source mapping
    span_start: int = 0
    span_end: int = 0
    
    # Page info
    page_numbers: list[int] = field(default_factory=list)
    
    # Element tracking
    element_indices: list[int] = field(default_factory=list)
    element_kinds: list[str] = field(default_factory=list)
    
    # Overlap info
    overlap_start: int = 0  # chars from previous chunk
    overlap_end: int = 0    # chars into next chunk
    
    # Content flags
    has_table: bool = False
    has_figure: bool = False
    has_code: bool = False
    has_formula: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        data = asdict(self)
        # These fields are only useful internally and should not be in the JSON payload.
        excluded = {
            "span_start",
            "span_end",
            "page_numbers",
            "element_indices",
            "element_kinds",
            "overlap_start",
            "overlap_end",
            "has_table",
            "has_figure",
            "has_code",
            "has_formula",
        }
        for key in excluded:
            data.pop(key, None)
        return data


@dataclass
class ChunkerConfig:
    """Configuration for the chunker."""
    max_chars: int = 4000
    min_chunk_chars: int = 800
    overlap_chars: int = 200
    preserve_lists: bool = True
    preserve_tables: bool = True
    preserve_figures: bool = True
    include_section_header_in_chunk: bool = True
    semantic_boundary_patterns: list[str] = field(default_factory=lambda: [
        r"\.\s+",      # Sentence end
        r"\n\n",       # Paragraph break
        r"\n(?=[-*•])", # List start
        r"\n(?=#)",    # Heading start
    ])


# =============================================================================
# Span and Page Utilities
# =============================================================================

def compute_span(spans) -> Optional[tuple[int, int]]:
    """Compute [start, end) from DI spans collection."""
    if not spans:
        return None
    offsets = [s.offset for s in spans]
    lengths = [s.length for s in spans]
    start = min(offsets)
    end = max(o + l for o, l in zip(offsets, lengths))
    return start, end


def collect_pages(obj) -> list[int]:
    """Extract page numbers from bounding_regions."""
    pages: set[int] = set()
    brs = getattr(obj, "bounding_regions", None)
    if brs:
        for br in brs:
            pn = getattr(br, "page_number", None)
            if pn is not None:
                pages.add(pn)
    return sorted(pages)


def infer_heading_level(markdown: str, start: int, end: int) -> Optional[int]:
    """Infer heading level from markdown # syntax."""
    segment = markdown[start:end].lstrip()
    if not segment.startswith("#"):
        return None
    hashes = len(segment) - len(segment.lstrip("#"))
    return hashes if 0 < hashes <= 6 else None


def extract_heading_text(markdown: str, el: NormalizedElement) -> str:
    """Extract clean heading text without # markers."""
    seg = markdown[el.span_start:el.span_end].strip()
    if seg.startswith("#"):
        seg = seg.lstrip("#").lstrip()
    return seg


def is_list_item(markdown: str, start: int, end: int) -> bool:
    """Check if this span is a list item."""
    segment = markdown[start:end].lstrip()
    # Markdown list patterns: -, *, +, or numbered (1., 2., etc.)
    return bool(re.match(r"^[-*+]|\d+\.\s", segment))


def find_semantic_boundary(
    text: str, 
    target_pos: int, 
    search_range: int = 200,
    patterns: list[str] | None = None
) -> int:
    """
    Find the best semantic boundary near target_pos.
    Looks for sentence ends, paragraph breaks, etc.
    """
    if patterns is None:
        patterns = [r"\.\s+", r"\n\n", r"\n"]
    
    start = max(0, target_pos - search_range)
    end = min(len(text), target_pos + search_range)
    search_text = text[start:end]
    
    best_pos = target_pos
    best_distance = search_range + 1
    
    for pattern in patterns:
        for match in re.finditer(pattern, search_text):
            abs_pos = start + match.end()
            distance = abs(abs_pos - target_pos)
            if distance < best_distance:
                best_distance = distance
                best_pos = abs_pos
    
    return best_pos


# =============================================================================
# Element Normalization
# =============================================================================

def normalize_elements(result, markdown: str, config: ChunkerConfig) -> list[NormalizedElement]:
    """
    Convert Azure DI result into normalized elements with unified spans.
    Handles paragraphs, tables, figures, sections, and detects lists.
    """
    elements: list[NormalizedElement] = []
    list_group_counter = 0
    prev_was_list = False

    # Process paragraphs
    for p in getattr(result, "paragraphs", []) or []:
        span = compute_span(p.spans)
        if not span:
            continue
        
        start, end = span
        role = getattr(p, "role", None)
        heading_level = None
        is_list = is_list_item(markdown, start, end)
        
        # Detect headings
        if role in HEADING_ROLES or role == "sectionHeading":
            heading_level = infer_heading_level(markdown, start, end)
        
        # List grouping for preservation
        list_group = None
        if config.preserve_lists and is_list:
            if not prev_was_list:
                list_group_counter += 1
            list_group = f"list_{list_group_counter}"
        prev_was_list = is_list
        
        elements.append(NormalizedElement(
            kind=ElementKind.PARAGRAPH,
            span_start=start,
            span_end=end,
            role=role,
            heading_level=heading_level,
            pages=collect_pages(p),
            is_atomic=False,
            is_list_item=is_list,
            list_group_id=list_group,
            raw=p,
        ))

    # Process tables (atomic - never split)
    for t in getattr(result, "tables", []) or []:
        span = compute_span(t.spans)
        if not span:
            continue
        
        elements.append(NormalizedElement(
            kind=ElementKind.TABLE,
            span_start=span[0],
            span_end=span[1],
            pages=collect_pages(t),
            is_atomic=config.preserve_tables,
            raw=t,
        ))

    # Process figures (atomic)
    for f in getattr(result, "figures", []) or []:
        span = compute_span(f.spans)
        if not span:
            continue
        
        elements.append(NormalizedElement(
            kind=ElementKind.FIGURE,
            span_start=span[0],
            span_end=span[1],
            pages=collect_pages(f),
            is_atomic=config.preserve_figures,
            raw=f,
        ))

    # Process sections (for additional structure info)
    for s in getattr(result, "sections", []) or []:
        span = compute_span(s.spans)
        if not span:
            continue
        
        level = getattr(s, "level", None) or getattr(s, "heading_level", None)
        elements.append(NormalizedElement(
            kind=ElementKind.SECTION,
            span_start=span[0],
            span_end=span[1],
            heading_level=level,
            pages=collect_pages(s),
            raw=s,
        ))

    # Detect code blocks from markdown
    code_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    for match in code_pattern.finditer(markdown):
        elements.append(NormalizedElement(
            kind=ElementKind.CODE,
            span_start=match.start(),
            span_end=match.end(),
            is_atomic=True,
        ))

    # Detect LaTeX formulas
    formula_patterns = [
        re.compile(r"\$\$[\s\S]*?\$\$"),  # Display math
        re.compile(r"\\begin\{equation\}[\s\S]*?\\end\{equation\}"),
    ]
    for pattern in formula_patterns:
        for match in pattern.finditer(markdown):
            elements.append(NormalizedElement(
                kind=ElementKind.FORMULA,
                span_start=match.start(),
                span_end=match.end(),
                is_atomic=True,
            ))

    # Sort by span_start for reading order
    elements.sort(key=lambda e: (e.span_start, -e.span_end))
    
    # Deduplicate overlapping elements (prefer more specific)
    elements = _deduplicate_elements(elements)
    
    logger.info(f"Normalized {len(elements)} elements from document")
    return elements


def _deduplicate_elements(elements: list[NormalizedElement]) -> list[NormalizedElement]:
    """Remove duplicate/overlapping elements, preferring more specific ones."""
    if not elements:
        return elements
    
    result = []
    seen_spans: set[tuple[int, int]] = set()
    
    for el in elements:
        span_key = (el.span_start, el.span_end)
        if span_key in seen_spans:
            continue
        
        # Skip section elements that fully contain other elements
        if el.kind == ElementKind.SECTION:
            has_overlap = any(
                other.span_start >= el.span_start and 
                other.span_end <= el.span_end and
                other is not el and
                other.kind != ElementKind.SECTION
                for other in elements
            )
            if has_overlap:
                continue
        
        seen_spans.add(span_key)
        result.append(el)
    
    return result


# =============================================================================
# Hierarchical Section Building
# =============================================================================

def build_section_hierarchy(
    elements: list[NormalizedElement], 
    markdown: str
) -> list[SectionNode]:
    """
    Build hierarchical sections with proper parent tracking.
    Returns flat list but with parent pointers for breadcrumb generation.
    """
    sections: list[SectionNode] = []
    section_stack: list[SectionNode] = []  # Stack for hierarchy tracking
    
    # Create root section for preamble content
    root = SectionNode(title=None, level=0, parent=None)
    current_section = root
    
    for idx, el in enumerate(elements):
        is_heading = (
            el.kind == ElementKind.PARAGRAPH and
            el.heading_level is not None
        )
        
        if is_heading:
            # Finish current section
            if current_section.element_indices or current_section.title:
                sections.append(current_section)
            
            # Pop stack until we find a parent with lower level
            while section_stack and section_stack[-1].level >= el.heading_level:
                section_stack.pop()
            
            # Create new section with proper parent
            parent = section_stack[-1] if section_stack else None
            title = extract_heading_text(markdown, el)
            
            new_section = SectionNode(
                title=title,
                level=el.heading_level,
                element_indices=[idx],
                parent=parent,
            )
            
            section_stack.append(new_section)
            current_section = new_section
        else:
            current_section.element_indices.append(idx)
    
    # Don't forget the last section
    if current_section.element_indices or current_section.title:
        sections.append(current_section)
    
    logger.info(f"Built {len(sections)} sections with hierarchy")
    return sections


# =============================================================================
# Chunk Building
# =============================================================================

def build_chunks(
    sections: list[SectionNode],
    elements: list[NormalizedElement],
    markdown: str,
    doc_id: str,
    config: ChunkerConfig,
) -> list[Chunk]:
    """
    Build chunks from sections, respecting element boundaries and config.
    """
    all_chunks: list[Chunk] = []
    
    for sec_idx, section in enumerate(sections):
        section_chunks = _chunk_section(
            section=section,
            elements=elements,
            markdown=markdown,
            doc_id=doc_id,
            section_index=sec_idx,
            config=config,
        )
        all_chunks.extend(section_chunks)
    
    # Apply overlap between chunks
    if config.overlap_chars > 0:
        all_chunks = _apply_overlap(all_chunks, markdown, config)
    
    logger.info(f"Created {len(all_chunks)} chunks")
    return all_chunks


def _chunk_section(
    section: SectionNode,
    elements: list[NormalizedElement],
    markdown: str,
    doc_id: str,
    section_index: int,
    config: ChunkerConfig,
) -> list[Chunk]:
    """Chunk a single section, respecting element boundaries."""
    chunks: list[Chunk] = []
    
    if not section.element_indices:
        return chunks
    
    # Group list items together
    element_groups = _group_list_items(section.element_indices, elements)
    
    # Build chunks from groups
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    current_elem_ids: list[int] = []
    current_pages: set[int] = set()
    current_kinds: set[ElementKind] = set()
    
    def flush_chunk():
        nonlocal current_start, current_end, current_elem_ids, current_pages, current_kinds
        
        if current_start is None or current_end is None:
            return
        
        text = markdown[current_start:current_end].strip()
        if not text:
            return
        
        # Add section header prefix if configured
        if config.include_section_header_in_chunk and section.title and chunks == []:
            breadcrumb = section.get_breadcrumb_str()
            if breadcrumb:
                text = f"[{breadcrumb}]\n\n{text}"
        
        chunk_id = f"{doc_id}::s{section_index}::c{len(chunks)}"
        
        chunk = Chunk(
            id=chunk_id,
            doc_id=doc_id,
            text=text,
            section_title=section.title,
            section_level=section.level,
            section_breadcrumb=section.get_breadcrumb(),
            span_start=current_start,
            span_end=current_end,
            page_numbers=sorted(current_pages),
            element_indices=current_elem_ids.copy(),
            element_kinds=[k.value for k in current_kinds],
            has_table=ElementKind.TABLE in current_kinds,
            has_figure=ElementKind.FIGURE in current_kinds,
            has_code=ElementKind.CODE in current_kinds,
            has_formula=ElementKind.FORMULA in current_kinds,
        )
        chunks.append(chunk)
        
        # Reset
        current_start = None
        current_end = None
        current_elem_ids = []
        current_pages = set()
        current_kinds = set()
    
    for group in element_groups:
        # Calculate group span
        group_elements = [elements[idx] for idx in group]
        group_start = min(el.span_start for el in group_elements)
        group_end = max(el.span_end for el in group_elements)
        group_len = group_end - group_start
        group_pages = set()
        group_kinds = set()
        is_atomic = any(el.is_atomic for el in group_elements)
        
        for el in group_elements:
            group_pages.update(el.pages)
            group_kinds.add(el.kind)
        
        if current_start is None:
            # Start new chunk
            current_start = group_start
            current_end = group_end
            current_elem_ids = list(group)
            current_pages = group_pages
            current_kinds = group_kinds
            continue
        
        prospective_len = group_end - current_start
        current_len = current_end - current_start
        
        # Decision: flush and start new, or extend?
        should_flush = False
        
        if is_atomic and group_len > config.max_chars:
            # Atomic element exceeds max - flush current, then add as own chunk
            if current_len > 0:
                flush_chunk()
            current_start = group_start
            current_end = group_end
            current_elem_ids = list(group)
            current_pages = group_pages
            current_kinds = group_kinds
            flush_chunk()
            continue
        
        if prospective_len > config.max_chars:
            if current_len >= config.min_chunk_chars:
                should_flush = True
            elif is_atomic:
                # Current is small but next is atomic - flush anyway
                should_flush = True
        
        if should_flush:
            flush_chunk()
            current_start = group_start
            current_end = group_end
            current_elem_ids = list(group)
            current_pages = group_pages
            current_kinds = group_kinds
        else:
            # Extend current chunk
            current_end = group_end
            current_elem_ids.extend(group)
            current_pages.update(group_pages)
            current_kinds.update(group_kinds)
    
    # Flush remaining
    flush_chunk()
    
    return chunks


def _group_list_items(
    element_indices: list[int], 
    elements: list[NormalizedElement]
) -> list[list[int]]:
    """Group consecutive list items together."""
    groups: list[list[int]] = []
    current_group: list[int] = []
    current_list_id: Optional[str] = None
    
    for idx in element_indices:
        el = elements[idx]
        
        if el.list_group_id:
            if el.list_group_id == current_list_id:
                current_group.append(idx)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [idx]
                current_list_id = el.list_group_id
        else:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_list_id = None
            groups.append([idx])
    
    if current_group:
        groups.append(current_group)
    
    return groups


def _apply_overlap(
    chunks: list[Chunk], 
    markdown: str, 
    config: ChunkerConfig
) -> list[Chunk]:
    """Apply overlap between consecutive chunks for context continuity."""
    if len(chunks) < 2:
        return chunks
    
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        curr_chunk = chunks[i]
        
        # Calculate overlap from previous chunk
        overlap_start = max(
            prev_chunk.span_end - config.overlap_chars,
            prev_chunk.span_start
        )
        
        # Find semantic boundary for overlap
        overlap_start = find_semantic_boundary(
            markdown, 
            overlap_start, 
            search_range=50,
            patterns=config.semantic_boundary_patterns
        )
        
        overlap_text = markdown[overlap_start:prev_chunk.span_end].strip()
        
        if overlap_text:
            # Prepend overlap to current chunk
            curr_chunk.text = f"...{overlap_text}\n\n{curr_chunk.text}"
            curr_chunk.overlap_start = len(overlap_text)
            prev_chunk.overlap_end = len(overlap_text)
    
    return chunks


# =============================================================================
# Main Pipeline
# =============================================================================

def process_document(
    pdf_path: str | Path,
    endpoint: str,
    api_key: str,
    doc_id: Optional[str] = None,
    config: Optional[ChunkerConfig] = None,
) -> list[Chunk]:
    """
    Full pipeline: Analyze PDF with Azure DI, then chunk with structure preservation.
    """
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import (
            AnalyzeDocumentRequest,
            DocumentContentFormat,
            AnalyzeOutputOption,
        )
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        logger.error("Azure Document Intelligence SDK not installed.")
        logger.error("Install with: pip install azure-ai-documentintelligence")
        raise
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    if doc_id is None:
        doc_id = pdf_path.stem or str(uuid.uuid4())[:8]
    
    if config is None:
        config = ChunkerConfig()
    
    logger.info(f"Processing document: {pdf_path}")
    logger.info(f"Document ID: {doc_id}")
    
    # Initialize client
    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )
    
    # Analyze document
    logger.info("Sending document to Azure Document Intelligence...")
    
    with open(pdf_path, "rb") as f:
        analyze_kwargs = dict(
            model_id="prebuilt-layout",
            body=AnalyzeDocumentRequest(bytes_source=f.read()),
            output_content_format=DocumentContentFormat.MARKDOWN,
            string_index_type="unicodeCodePoint",  # Safe for Python slicing
        )

        # The current Azure DI SDK (v1.x) only supports requesting FIGURES/PDF via `output`.
        # Pages, tables, and paragraphs are always returned for layout analysis, so we only
        # ask for figures when needed to avoid unsupported enum errors like "PAGES".
        output_options = []
        if config.preserve_figures:
            output_options.append(AnalyzeOutputOption.FIGURES)
        if output_options:
            analyze_kwargs["output"] = output_options

        poller = client.begin_analyze_document(**analyze_kwargs)
    
    result = poller.result()
    logger.info("Document analysis complete")
    
    # Get markdown content
    markdown = getattr(result, "content", None)
    if not markdown:
        raise ValueError("No markdown content returned from Azure DI")
    
    logger.info(f"Markdown content length: {len(markdown)} chars")
    
    # Normalize elements
    elements = normalize_elements(result, markdown, config)
    
    if not elements:
        logger.warning("No elements found, creating single chunk from markdown")
        return [Chunk(
            id=f"{doc_id}::s0::c0",
            doc_id=doc_id,
            text=markdown.strip(),
            span_start=0,
            span_end=len(markdown),
        )]
    
    # Build section hierarchy
    sections = build_section_hierarchy(elements, markdown)
    
    # Build chunks
    chunks = build_chunks(sections, elements, markdown, doc_id, config)
    
    return chunks


def chunks_to_json(chunks: list[Chunk], output_path: str | Path) -> None:
    """Save chunks to JSON file."""
    output_path = Path(output_path)
    data = [c.to_dict() for c in chunks]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def print_chunk_summary(chunks: list[Chunk]) -> None:
    """Print summary of chunks."""
    print("\n" + "=" * 60)
    print(f"CHUNK SUMMARY: {len(chunks)} chunks")
    print("=" * 60)
    
    total_chars = sum(len(c.text) for c in chunks)
    avg_chars = total_chars / len(chunks) if chunks else 0
    
    print(f"Total characters: {total_chars:,}")
    print(f"Average chunk size: {avg_chars:,.0f} chars")
    
    tables = sum(1 for c in chunks if c.has_table)
    figures = sum(1 for c in chunks if c.has_figure)
    code = sum(1 for c in chunks if c.has_code)
    
    print(f"Chunks with tables: {tables}")
    print(f"Chunks with figures: {figures}")
    print(f"Chunks with code: {code}")
    
    print("\n" + "-" * 60)
    print("First 3 chunks preview:")
    print("-" * 60)
    
    for i, chunk in enumerate(chunks[:3]):
        breadcrumb = " > ".join(chunk.section_breadcrumb) if chunk.section_breadcrumb else "(root)"
        preview = chunk.text[:200].replace("\n", " ")
        if len(chunk.text) > 200:
            preview += "..."
        
        print(f"\n[{chunk.id}]")
        print(f"  Section: {breadcrumb}")
        print(f"  Pages: {chunk.page_numbers}")
        print(f"  Length: {len(chunk.text)} chars")
        print(f"  Preview: {preview}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Azure Document Intelligence chunker with structure preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf --api-key KEY --endpoint URL
  %(prog)s document.pdf --api-key KEY --endpoint URL --output chunks.json
  %(prog)s document.pdf --api-key KEY --endpoint URL --max-chars 2000 --overlap 300
        """,
    )
    
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    parser.add_argument("--api-key", required=True, help="Azure DI API key")
    parser.add_argument("--endpoint", required=True, help="Azure DI endpoint URL")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file (default: <pdf_name>_chunks.json)")
    parser.add_argument("--doc-id", help="Document ID (default: PDF filename)")
    
    # Chunking config
    parser.add_argument("--max-chars", type=int, default=4000, help="Maximum chunk size in chars (default: 4000)")
    parser.add_argument("--min-chars", type=int, default=800, help="Minimum chunk size in chars (default: 800)")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks in chars (default: 200)")
    parser.add_argument("--no-overlap", action="store_true", help="Disable chunk overlap")
    parser.add_argument("--no-section-header", action="store_true", help="Don't include section headers in chunks")
    
    # Output options
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress summary output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Build config
    config = ChunkerConfig(
        max_chars=args.max_chars,
        min_chunk_chars=args.min_chars,
        overlap_chars=0 if args.no_overlap else args.overlap,
        include_section_header_in_chunk=not args.no_section_header,
    )
    
    try:
        # Process document
        chunks = process_document(
            pdf_path=args.pdf,
            endpoint=args.endpoint,
            api_key=args.api_key,
            doc_id=args.doc_id,
            config=config,
        )
        
        # Output
        output_path = args.output or args.pdf.with_suffix("").with_suffix("_chunks.json")
        chunks_to_json(chunks, output_path)
        
        if not args.quiet:
            print_chunk_summary(chunks)
            print(f"\nOutput saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

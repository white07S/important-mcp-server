#!/usr/bin/env python3
"""
Quick usage example for ECharts Schema tools.

Step 1: Build the index (one-time, ~10-30 seconds)
    python echarts_schema_indexer.py schema.json echarts.index.json

Step 2: Use the explorer (fast, <100ms per query)
    python echarts_explorer.py echarts.index.json --list-series
    python echarts_explorer.py echarts.index.json --explore-series line
    python echarts_explorer.py echarts.index.json --validate my_chart.json
"""

from explorer import EChartsExplorer, ValidationResult


def demo_exploration(explorer: EChartsExplorer):
    """Demonstrate exploration capabilities."""
    
    print("=" * 60)
    print("EXPLORATION DEMO")
    print("=" * 60)
    
    # List all series types
    series_types = explorer.list_series_types()
    print(f"\n📊 Found {len(series_types)} series types:")
    print(f"   {', '.join(series_types[:10])}{'...' if len(series_types) > 10 else ''}")
    
    # Explore a specific series
    print("\n" + "-" * 40)
    print("Exploring 'line' series:")
    print(explorer.get_series_summary("line"))
    
    # Search functionality
    print("\n" + "-" * 40)
    print("Searching for 'smooth':")
    results = explorer.search("smooth")
    for type_, path in results[:5]:
        print(f"  [{type_}] {path}")


def demo_validation(explorer: EChartsExplorer):
    """Demonstrate validation capabilities."""
    
    print("\n" + "=" * 60)
    print("VALIDATION DEMO")
    print("=" * 60)
    
    # Valid configuration
    valid_config = {
        "type": "line",
        "data": [1, 2, 3, 4, 5],
        "smooth": True,
        "name": "Sales"
    }
    
    print("\n✓ Validating correct config:")
    result = explorer.validate_series_config("line", valid_config)
    print(f"  {result}")
    
    # Config with issues
    issue_config = {
        "type": "line",
        "data": [1, 2, 3],
        "smooth": "yes",  # Should be boolean
        "unknownProp": 123
    }
    
    print("\n⚠ Validating config with issues:")
    result = explorer.validate_series_config("line", issue_config)
    print(f"  {result}")
    
    # Full option validation
    full_option = {
        "title": {"text": "My Chart"},
        "xAxis": {"type": "category", "data": ["A", "B", "C"]},
        "yAxis": {"type": "value"},
        "series": [
            {"type": "bar", "data": [10, 20, 30]},
            {"type": "line", "data": [15, 25, 35], "smooth": True}
        ]
    }
    
    print("\n📋 Validating full option:")
    result = explorer.validate_option(full_option)
    print(f"  {result}")


def demo_mcp_export(explorer: EChartsExplorer):
    """Demonstrate MCP catalog export."""
    
    print("\n" + "=" * 60)
    print("MCP EXPORT DEMO")
    print("=" * 60)
    
    catalog = explorer.export_mcp_catalog()
    
    print(f"\n📦 MCP Catalog generated:")
    print(f"   Chart types: {len(catalog['chart_types'])}")
    print(f"   Components: {len(catalog['components'])}")
    
    # Show sample entry
    if "line" in catalog["chart_types"]:
        line = catalog["chart_types"]["line"]
        print(f"\n   Sample entry (line):")
        print(f"   - Properties: {line['property_count']}")
        key_props = list(line['key_properties'].keys())[:5]
        print(f"   - Key props: {', '.join(key_props)}...")


def main():
    index_path = "/Users/preetam/Develop/mcp_servers/viz-mcp/process/schema.index.json"
    print(f"Loading index from: {index_path}")
    explorer = EChartsExplorer(index_path)
    
    demo_exploration(explorer)
    demo_validation(explorer)
    demo_mcp_export(explorer)
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
ECharts Schema Indexer
Pre-processes the large ECharts schema JSON into an efficient indexed structure.
Run this once to generate the index file, then use echarts_explorer.py for fast lookups.
"""

import json
import re
from pathlib import Path
from typing import Any
from collections import defaultdict


def strip_html(text: str) -> str:
    """Remove HTML tags and clean up description text."""
    if not text:
        return ""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    # Remove code blocks
    clean = re.sub(r'```[\s\S]*?```', '', clean)
    # Collapse whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Truncate if too long
    return clean[:500] if len(clean) > 500 else clean


def extract_type_info(type_val: Any) -> list[str]:
    """Extract type information as a list of strings."""
    if isinstance(type_val, list):
        return [str(t) for t in type_val]
    elif isinstance(type_val, str):
        return [type_val]
    return ["unknown"]


def extract_property_info(prop_data: dict, max_depth: int = 3) -> dict:
    """Extract essential property information."""
    info = {
        "type": extract_type_info(prop_data.get("type", ["unknown"])),
        "description": strip_html(prop_data.get("description", "")),
    }
    
    if "default" in prop_data:
        info["default"] = prop_data["default"]
    
    if "uiControl" in prop_data:
        ui = prop_data["uiControl"]
        if isinstance(ui, dict):
            if "options" in ui:
                info["options"] = ui["options"].split(",") if isinstance(ui["options"], str) else ui["options"]
            if "min" in ui:
                info["min"] = ui["min"]
            if "max" in ui:
                info["max"] = ui["max"]
    
    # Extract nested properties (limited depth)
    if max_depth > 0 and "properties" in prop_data:
        info["properties"] = {}
        for key, val in prop_data["properties"].items():
            if isinstance(val, dict):
                info["properties"][key] = extract_property_info(val, max_depth - 1)
    
    return info


def extract_series_types(schema: dict) -> dict:
    """Extract all series types and their configurations."""
    series_types = {}
    
    option = schema.get("option", {})
    properties = option.get("properties", {})
    series = properties.get("series", {})
    
    # Series can be defined in different ways in the schema
    # Look for items or anyOf patterns
    series_items = series.get("items", {})
    
    if "anyOf" in series_items:
        for item in series_items["anyOf"]:
            if "properties" in item:
                props = item["properties"]
                if "type" in props:
                    type_info = props["type"]
                    # Get the default value which indicates the series type
                    series_type = type_info.get("default", "").strip("'\"")
                    if series_type:
                        series_types[series_type] = {
                            "description": strip_html(item.get("description", "")),
                            "properties": {}
                        }
                        for key, val in props.items():
                            if isinstance(val, dict):
                                series_types[series_type]["properties"][key] = extract_property_info(val, max_depth=2)
    
    # Also check for direct properties pattern
    if "properties" in series:
        for key, val in series["properties"].items():
            if isinstance(val, dict) and "properties" in val:
                series_types[key] = {
                    "description": strip_html(val.get("description", "")),
                    "properties": {}
                }
                for prop_key, prop_val in val["properties"].items():
                    if isinstance(prop_val, dict):
                        series_types[key]["properties"][prop_key] = extract_property_info(prop_val, max_depth=2)
    
    return series_types


def extract_components(schema: dict) -> dict:
    """Extract all top-level components (xAxis, yAxis, tooltip, etc.)."""
    components = {}
    
    option = schema.get("option", {})
    properties = option.get("properties", {})
    
    # Key components to extract
    component_keys = [
        "title", "legend", "grid", "xAxis", "yAxis", "polar", "radiusAxis",
        "angleAxis", "radar", "dataZoom", "visualMap", "tooltip", "axisPointer",
        "toolbox", "brush", "geo", "parallel", "parallelAxis", "singleAxis",
        "timeline", "graphic", "calendar", "dataset", "aria", "series"
    ]
    
    for key in component_keys:
        if key in properties:
            comp = properties[key]
            if isinstance(comp, dict):
                components[key] = {
                    "type": extract_type_info(comp.get("type", ["unknown"])),
                    "description": strip_html(comp.get("description", "")),
                    "properties": {}
                }
                
                # Handle array types (items)
                if "items" in comp:
                    items = comp["items"]
                    if isinstance(items, dict) and "properties" in items:
                        for prop_key, prop_val in items["properties"].items():
                            if isinstance(prop_val, dict):
                                components[key]["properties"][prop_key] = extract_property_info(prop_val, max_depth=2)
                
                # Handle direct properties
                if "properties" in comp:
                    for prop_key, prop_val in comp["properties"].items():
                        if isinstance(prop_val, dict):
                            components[key]["properties"][prop_key] = extract_property_info(prop_val, max_depth=2)
    
    return components


def build_search_index(components: dict, series_types: dict) -> dict:
    """Build a keyword search index for fast lookups."""
    index = defaultdict(list)
    
    # Index components
    for comp_name, comp_data in components.items():
        index[comp_name.lower()].append(("component", comp_name))
        
        for prop_name in comp_data.get("properties", {}):
            index[prop_name.lower()].append(("component_property", f"{comp_name}.{prop_name}"))
    
    # Index series types
    for series_name, series_data in series_types.items():
        index[series_name.lower()].append(("series", series_name))
        
        for prop_name in series_data.get("properties", {}):
            index[prop_name.lower()].append(("series_property", f"{series_name}.{prop_name}"))
    
    return dict(index)


def extract_all_series_from_schema(schema: dict) -> dict:
    """Deep extraction of series types by traversing the entire schema."""
    series_types = {}
    
    def find_series_definitions(obj: Any, path: str = "") -> None:
        if not isinstance(obj, dict):
            return
            
        # Check if this looks like a series type definition
        if "type" in obj and isinstance(obj.get("type"), dict):
            type_prop = obj["type"]
            if "default" in type_prop:
                default_val = str(type_prop["default"]).strip("'\"")
                if default_val and default_val not in series_types:
                    series_types[default_val] = {
                        "description": strip_html(obj.get("description", "")),
                        "properties": {},
                        "path": path
                    }
                    if "properties" in obj:
                        for key, val in obj["properties"].items():
                            if isinstance(val, dict):
                                series_types[default_val]["properties"][key] = extract_property_info(val, max_depth=2)
        
        # Recurse into nested structures
        for key, val in obj.items():
            if isinstance(val, dict):
                find_series_definitions(val, f"{path}.{key}" if path else key)
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, dict):
                        find_series_definitions(item, f"{path}.{key}[{i}]" if path else f"{key}[{i}]")
    
    find_series_definitions(schema)
    return series_types


def create_index(schema_path: str, output_path: str = None) -> dict:
    """
    Create an optimized index from the ECharts schema.
    
    Args:
        schema_path: Path to the full ECharts schema JSON
        output_path: Optional path to save the index (defaults to schema_path with .index.json suffix)
    
    Returns:
        The created index dictionary
    """
    print(f"Loading schema from {schema_path}...")
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    print("Extracting components...")
    components = extract_components(schema)
    
    print("Extracting series types...")
    series_types = extract_series_types(schema)
    
    # Fallback to deep extraction if the main extractor fails (schema variants)
    if not series_types:
        print("Primary extraction yielded no series types. Falling back to deep search...")
        series_types = extract_all_series_from_schema(schema)
    
    print("Building search index...")
    search_index = build_search_index(components, series_types)
    
    # Create the final index
    index = {
        "version": "1.0",
        "source": schema_path,
        "components": components,
        "series_types": series_types,
        "search_index": search_index,
        "series_list": sorted(series_types.keys()),
        "component_list": sorted(components.keys())
    }
    
    # Save to file
    if output_path is None:
        output_path = str(Path(schema_path).with_suffix('.index.json'))
    
    print(f"Saving index to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Index size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Found {len(series_types)} series types and {len(components)} components")
    
    return index


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python echarts_schema_indexer.py <schema.json> [output.index.json]")
        print("Example: python echarts_schema_indexer.py schema.json echarts.index.json")
        sys.exit(1)
    
    schema_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    create_index(schema_path, output_path)

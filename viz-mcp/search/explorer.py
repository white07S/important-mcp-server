#!/usr/bin/env python3
"""
ECharts Schema Explorer & Validator
Fast exploration and validation using the pre-built index.

Usage:
    from echarts_explorer import EChartsExplorer
    
    explorer = EChartsExplorer("echarts.index.json")
    
    # Explore
    explorer.list_series_types()
    explorer.explore_series("line")
    explorer.explore_component("xAxis")
    explorer.search("smooth")
    
    # Validate
    explorer.validate_series_config("line", {"smooth": True, "data": [1,2,3]})
"""

import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of schema validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unknown_properties: list[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.valid:
            msg = "✓ Valid"
            if self.warnings:
                msg += f"\n  Warnings: {', '.join(self.warnings)}"
            if self.unknown_properties:
                msg += f"\n  Unknown properties (may be valid): {', '.join(self.unknown_properties)}"
            return msg
        else:
            return f"✗ Invalid\n  Errors: {', '.join(self.errors)}"


class EChartsExplorer:
    """
    Fast ECharts schema explorer and validator.
    Uses pre-built index for O(1) lookups.
    """
    
    def __init__(self, index_path: str):
        """
        Initialize with pre-built index file.
        
        Args:
            index_path: Path to the .index.json file created by echarts_schema_indexer.py
        """
        self._index_path = Path(index_path)
        self._index: Optional[dict] = None
        self._loaded = False
    
    def _ensure_loaded(self) -> None:
        """Lazy load the index on first use."""
        if not self._loaded:
            with open(self._index_path, 'r', encoding='utf-8') as f:
                self._index = json.load(f)
            self._loaded = True
    
    @property
    def index(self) -> dict:
        self._ensure_loaded()
        return self._index
    
    # ==================== EXPLORATION METHODS ====================
    
    def list_series_types(self) -> list[str]:
        """Get all available series types."""
        return self.index.get("series_list", [])
    
    def list_components(self) -> list[str]:
        """Get all available components."""
        return self.index.get("component_list", [])
    
    def explore_series(self, series_type: str) -> Optional[dict]:
        """
        Get detailed information about a series type.
        
        Args:
            series_type: e.g., "line", "bar", "pie", "scatter"
        
        Returns:
            Dictionary with description and properties, or None if not found
        """
        series_types = self.index.get("series_types", {})
        return series_types.get(series_type)
    
    def explore_component(self, component: str) -> Optional[dict]:
        """
        Get detailed information about a component.
        
        Args:
            component: e.g., "xAxis", "yAxis", "tooltip", "legend"
        
        Returns:
            Dictionary with description and properties, or None if not found
        """
        components = self.index.get("components", {})
        return components.get(component)
    
    def get_series_properties(self, series_type: str) -> dict:
        """Get all properties for a series type."""
        series = self.explore_series(series_type)
        if series:
            return series.get("properties", {})
        return {}
    
    def get_component_properties(self, component: str) -> dict:
        """Get all properties for a component."""
        comp = self.explore_component(component)
        if comp:
            return comp.get("properties", {})
        return {}
    
    def get_property_info(self, series_type: str, property_name: str) -> Optional[dict]:
        """Get information about a specific property of a series type."""
        props = self.get_series_properties(series_type)
        return props.get(property_name)
    
    def search(self, keyword: str, limit: int = 20) -> list[tuple[str, str]]:
        """
        Search for components/properties by keyword.
        
        Args:
            keyword: Search term
            limit: Maximum results to return
        
        Returns:
            List of (type, path) tuples
        """
        search_index = self.index.get("search_index", {})
        keyword_lower = keyword.lower()
        
        results = []
        
        # Exact match
        if keyword_lower in search_index:
            results.extend(search_index[keyword_lower])
        
        # Partial match
        for key, values in search_index.items():
            if keyword_lower in key and key != keyword_lower:
                results.extend(values)
            if len(results) >= limit:
                break
        
        return results[:limit]
    
    def get_series_summary(self, series_type: str) -> str:
        """Get a human-readable summary of a series type."""
        series = self.explore_series(series_type)
        if not series:
            return f"Series type '{series_type}' not found"
        
        lines = [f"Series: {series_type}"]
        if series.get("description"):
            lines.append(f"Description: {series['description'][:200]}...")
        
        props = series.get("properties", {})
        if props:
            lines.append(f"\nKey Properties ({len(props)} total):")
            # Show most important properties first
            priority_props = ["data", "type", "name", "itemStyle", "label", "emphasis", 
                           "smooth", "stack", "areaStyle", "symbol", "symbolSize"]
            
            shown = set()
            for prop in priority_props:
                if prop in props:
                    prop_info = props[prop]
                    type_str = ", ".join(prop_info.get("type", ["?"]))
                    default = prop_info.get("default", "")
                    default_str = f" (default: {default})" if default != "" else ""
                    lines.append(f"  • {prop}: {type_str}{default_str}")
                    shown.add(prop)
            
            # Show remaining (up to 10 more)
            remaining = [p for p in props if p not in shown][:10]
            for prop in remaining:
                prop_info = props[prop]
                type_str = ", ".join(prop_info.get("type", ["?"]))
                lines.append(f"  • {prop}: {type_str}")
            
            if len(props) > len(shown) + 10:
                lines.append(f"  ... and {len(props) - len(shown) - 10} more properties")
        
        return "\n".join(lines)
    
    # ==================== VALIDATION METHODS ====================
    
    def validate_series_config(self, series_type: str, config: dict) -> ValidationResult:
        """
        Validate a series configuration against the schema.
        
        Args:
            series_type: The series type (e.g., "line", "bar")
            config: The configuration dictionary to validate
        
        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(valid=True)
        
        # Check if series type exists
        series = self.explore_series(series_type)
        if not series:
            result.valid = False
            result.errors.append(f"Unknown series type: {series_type}")
            return result
        
        schema_props = series.get("properties", {})
        
        for key, value in config.items():
            if key not in schema_props:
                result.unknown_properties.append(key)
                continue
            
            prop_schema = schema_props[key]
            prop_types = prop_schema.get("type", [])
            
            # Type validation
            type_valid = self._validate_type(value, prop_types)
            if not type_valid:
                result.warnings.append(
                    f"Property '{key}' has value of type {type(value).__name__}, "
                    f"expected {prop_types}"
                )
            
            # Options validation (if enum-like)
            if "options" in prop_schema:
                valid_options = prop_schema["options"]
                if isinstance(value, str) and value not in valid_options:
                    result.warnings.append(
                        f"Property '{key}' value '{value}' not in valid options: {valid_options}"
                    )
            
            # Range validation
            if "min" in prop_schema and isinstance(value, (int, float)):
                min_val = float(prop_schema["min"])
                if value < min_val:
                    result.warnings.append(f"Property '{key}' value {value} is below minimum {min_val}")
            
            if "max" in prop_schema and isinstance(value, (int, float)):
                max_val = float(prop_schema["max"])
                if value > max_val:
                    result.warnings.append(f"Property '{key}' value {value} is above maximum {max_val}")
        
        return result
    
    def validate_option(self, option: dict) -> ValidationResult:
        """
        Validate a complete ECharts option configuration.
        
        Args:
            option: The full ECharts option dictionary
        
        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(valid=True)
        
        # Validate series
        if "series" in option:
            series_list = option["series"]
            if not isinstance(series_list, list):
                series_list = [series_list]
            
            for i, series_config in enumerate(series_list):
                if not isinstance(series_config, dict):
                    result.errors.append(f"series[{i}] is not a dictionary")
                    result.valid = False
                    continue
                
                series_type = series_config.get("type")
                if not series_type:
                    result.errors.append(f"series[{i}] missing 'type' property")
                    result.valid = False
                    continue
                
                series_result = self.validate_series_config(series_type, series_config)
                if not series_result.valid:
                    result.valid = False
                    result.errors.extend([f"series[{i}]: {e}" for e in series_result.errors])
                result.warnings.extend([f"series[{i}]: {w}" for w in series_result.warnings])
                result.unknown_properties.extend(
                    [f"series[{i}].{p}" for p in series_result.unknown_properties]
                )
        
        # Validate known components
        components = self.index.get("components", {})
        for comp_name in components:
            if comp_name in option and comp_name != "series":
                comp_config = option[comp_name]
                if isinstance(comp_config, list):
                    for i, item in enumerate(comp_config):
                        if isinstance(item, dict):
                            comp_result = self._validate_component_config(comp_name, item)
                            result.warnings.extend([f"{comp_name}[{i}]: {w}" for w in comp_result.warnings])
                elif isinstance(comp_config, dict):
                    comp_result = self._validate_component_config(comp_name, comp_config)
                    result.warnings.extend([f"{comp_name}: {w}" for w in comp_result.warnings])
        
        return result
    
    def _validate_component_config(self, component: str, config: dict) -> ValidationResult:
        """Validate a component configuration."""
        result = ValidationResult(valid=True)
        
        comp = self.explore_component(component)
        if not comp:
            return result
        
        schema_props = comp.get("properties", {})
        
        for key, value in config.items():
            if key not in schema_props:
                result.unknown_properties.append(key)
                continue
            
            prop_schema = schema_props[key]
            prop_types = prop_schema.get("type", [])
            
            type_valid = self._validate_type(value, prop_types)
            if not type_valid:
                result.warnings.append(
                    f"Property '{key}' has type {type(value).__name__}, expected {prop_types}"
                )
        
        return result
    
    def _validate_type(self, value: Any, expected_types: list[str]) -> bool:
        """Check if a value matches expected types."""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "boolean": bool,
            "Array": list,
            "Object": dict,
            "Function": str,  # Functions are often strings in JSON
            "Color": str,
            "*": object,  # Any type
        }
        
        for expected in expected_types:
            if expected in type_mapping:
                if isinstance(value, type_mapping[expected]):
                    return True
            elif expected == "null" and value is None:
                return True
        
        # Allow any value if types include Object or * or are unknown
        if "Object" in expected_types or "*" in expected_types:
            return True
        
        return False
    
    # ==================== EXPORT METHODS ====================
    
    def export_series_schema(self, series_type: str, output_path: str = None) -> dict:
        """
        Export a clean schema for a specific series type.
        Useful for MCP server tools.
        """
        series = self.explore_series(series_type)
        if not series:
            raise ValueError(f"Unknown series type: {series_type}")
        
        schema = {
            "series_type": series_type,
            "description": series.get("description", ""),
            "properties": series.get("properties", {})
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)
        
        return schema
    
    def export_mcp_catalog(self, output_path: str = None) -> dict:
        """
        Export a catalog optimized for MCP server usage.
        Includes all series types with their key properties.
        """
        catalog = {
            "chart_types": {},
            "components": self.list_components(),
        }
        
        for series_type in self.list_series_types():
            series = self.explore_series(series_type)
            if series:
                props = series.get("properties", {})
                
                # Extract key properties for quick reference
                key_props = {}
                for prop_name, prop_info in props.items():
                    key_props[prop_name] = {
                        "type": prop_info.get("type", ["unknown"]),
                    }
                    if "default" in prop_info:
                        key_props[prop_name]["default"] = prop_info["default"]
                    if "options" in prop_info:
                        key_props[prop_name]["options"] = prop_info["options"]
                
                catalog["chart_types"][series_type] = {
                    "description": series.get("description", "")[:300],
                    "property_count": len(props),
                    "key_properties": key_props
                }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(catalog, f, indent=2, ensure_ascii=False)
        
        return catalog


# ==================== CLI INTERFACE ====================

def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="ECharts Schema Explorer & Validator")
    parser.add_argument("index_path", help="Path to the .index.json file")
    parser.add_argument("--list-series", action="store_true", help="List all series types")
    parser.add_argument("--list-components", action="store_true", help="List all components")
    parser.add_argument("--explore-series", type=str, help="Explore a series type")
    parser.add_argument("--explore-component", type=str, help="Explore a component")
    parser.add_argument("--search", type=str, help="Search for a keyword")
    parser.add_argument("--validate", type=str, help="Validate a JSON config file")
    parser.add_argument("--export-mcp", type=str, help="Export MCP catalog to file")
    
    args = parser.parse_args()
    
    explorer = EChartsExplorer(args.index_path)
    
    if args.list_series:
        print("Available Series Types:")
        for s in explorer.list_series_types():
            print(f"  • {s}")
    
    elif args.list_components:
        print("Available Components:")
        for c in explorer.list_components():
            print(f"  • {c}")
    
    elif args.explore_series:
        print(explorer.get_series_summary(args.explore_series))
    
    elif args.explore_component:
        comp = explorer.explore_component(args.explore_component)
        if comp:
            print(f"Component: {args.explore_component}")
            print(f"Description: {comp.get('description', 'N/A')[:200]}")
            print(f"Properties: {len(comp.get('properties', {}))}")
            for prop, info in list(comp.get('properties', {}).items())[:15]:
                print(f"  • {prop}: {info.get('type', ['?'])}")
        else:
            print(f"Component '{args.explore_component}' not found")
    
    elif args.search:
        results = explorer.search(args.search)
        print(f"Search results for '{args.search}':")
        for type_, path in results:
            print(f"  [{type_}] {path}")
    
    elif args.validate:
        with open(args.validate, 'r') as f:
            config = json.load(f)
        result = explorer.validate_option(config)
        print(result)
    
    elif args.export_mcp:
        catalog = explorer.export_mcp_catalog(args.export_mcp)
        print(f"Exported MCP catalog with {len(catalog['chart_types'])} chart types")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
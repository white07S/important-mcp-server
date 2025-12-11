from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from search.explorer import EChartsExplorer, ValidationResult

INDEX_PATH = PROJECT_ROOT / "process" / "schema.index.json"


class ExplorerClient:
    """Manages the shared EChartsExplorer instance."""

    def __init__(self, index_path: Path):
        self._index_path = index_path
        self.explorer: EChartsExplorer | None = None

    async def ainit(self) -> None:
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"Schema index not found at {self._index_path}. Run process/index.py first."
            )
        explorer = EChartsExplorer(str(self._index_path))
        explorer.list_series_types()  # Prime cache
        self.explorer = explorer

    async def aclose(self) -> None:
        self.explorer = None

    def get_explorer(self) -> EChartsExplorer:
        if self.explorer is None:
            raise RuntimeError("Explorer not initialized yet.")
        return self.explorer


CLIENT: ExplorerClient | None = None


def _get_explorer() -> EChartsExplorer:
    if CLIENT is None:
        raise RuntimeError("Explorer client not ready.")
    return CLIENT.get_explorer()


@asynccontextmanager
async def lifespan(_server: FastMCP):
    global CLIENT
    CLIENT = ExplorerClient(INDEX_PATH)
    await CLIENT.ainit()
    try:
        yield {"index_path": str(INDEX_PATH)}
    finally:
        if CLIENT is not None:
            await CLIENT.aclose()
            CLIENT = None


mcp = FastMCP(
    name="viz-mcp",
    instructions=(
        "ECharts visualization server. Use resources to explore chart types and schemas. "
        "Use generate_chart tool to create validated ECharts option configs from data."
    ),
    lifespan=lifespan,
)


# =============================================================================
# RESOURCES - Schema exploration via MCP resources
# =============================================================================

@mcp.resource("echarts://charts")
def resource_list_charts() -> dict[str, Any]:
    """List all available ECharts chart types."""
    explorer = _get_explorer()
    series = explorer.list_series_types()
    components = explorer.list_components()
    return {
        "chart_types": series,
        "components": components,
        "chart_count": len(series),
        "component_count": len(components),
    }


@mcp.resource("echarts://charts/{chart_type}")
def resource_chart_schema(chart_type: str) -> dict[str, Any]:
    """Get schema details for a specific chart type."""
    explorer = _get_explorer()
    series = explorer.explore_series(chart_type)
    if not series:
        return {"error": f"Unknown chart type: {chart_type}", "available": explorer.list_series_types()}
    
    properties = series.get("properties", {})
    
    # Prioritize key properties
    priority = ["type", "data", "name", "id", "coordinateSystem", "smooth", "stack", 
                "areaStyle", "itemStyle", "label", "emphasis", "symbol", "symbolSize",
                "radius", "center", "roseType", "indicator", "radarIndex"]
    
    key_props = {}
    for prop in priority:
        if prop in properties:
            info = properties[prop]
            key_props[prop] = {
                "type": info.get("type"),
                "default": info.get("default"),
                "description": info.get("description", "")[:200],
            }
            if info.get("options"):
                key_props[prop]["options"] = info["options"]
    
    # Add remaining props
    for prop, info in properties.items():
        if prop not in key_props and len(key_props) < 40:
            key_props[prop] = {
                "type": info.get("type"),
                "default": info.get("default"),
            }
    
    return {
        "chart_type": chart_type,
        "description": series.get("description", ""),
        "property_count": len(properties),
        "properties": key_props,
    }


@mcp.resource("echarts://components/{component}")
def resource_component_schema(component: str) -> dict[str, Any]:
    """Get schema details for a component (xAxis, yAxis, tooltip, legend, etc.)."""
    explorer = _get_explorer()
    comp = explorer.explore_component(component)
    if not comp:
        return {"error": f"Unknown component: {component}", "available": explorer.list_components()}
    
    properties = comp.get("properties", {})
    props_subset = {}
    for i, (prop, info) in enumerate(properties.items()):
        if i >= 30:
            break
        props_subset[prop] = {
            "type": info.get("type"),
            "default": info.get("default"),
            "description": info.get("description", "")[:150],
        }
    
    return {
        "component": component,
        "description": comp.get("description", ""),
        "property_count": len(properties),
        "properties": props_subset,
    }


@mcp.resource("echarts://search/{keyword}")
def resource_search(keyword: str) -> dict[str, Any]:
    """Search schema for properties matching keyword."""
    explorer = _get_explorer()
    results = explorer.search(keyword, limit=20)
    return {
        "query": keyword,
        "matches": [{"kind": kind, "path": path} for kind, path in results],
        "count": len(results),
    }


# =============================================================================
# TOOLS - Minimal set for chart generation
# =============================================================================

@mcp.tool()
def ping() -> str:
    """Health check."""
    return "pong"


@mcp.tool()
def list_chart_types() -> dict[str, Any]:
    """
    List all available ECharts chart types.
    
    Returns:
        Dictionary with chart_types list and components list.
    """
    explorer = _get_explorer()
    return {
        "chart_types": explorer.list_series_types(),
        "components": explorer.list_components(),
    }


@mcp.tool()
def describe_chart(
    chart_type: Annotated[str, Field(description="Chart type to describe, e.g., 'line', 'bar', 'pie', 'radar'.")],
) -> dict[str, Any]:
    """
    Get detailed schema for a specific chart type including all configurable properties.
    
    Use this to understand what properties are available before building a chart config.
    
    Args:
        chart_type: The series type (line, bar, pie, scatter, radar, etc.)
    
    Returns:
        Schema with description and properties for the chart type.
    """
    explorer = _get_explorer()
    series = explorer.explore_series(chart_type)
    if not series:
        available = explorer.list_series_types()
        raise ToolError(f"Unknown chart type '{chart_type}'. Available: {', '.join(available[:20])}")
    
    properties = series.get("properties", {})
    
    # Format properties for readability
    formatted_props = {}
    for prop, info in properties.items():
        formatted_props[prop] = {
            "type": info.get("type"),
            "default": info.get("default"),
            "description": info.get("description", "")[:300],
        }
        if info.get("options"):
            formatted_props[prop]["options"] = info["options"]
    
    return {
        "chart_type": chart_type,
        "description": series.get("description", ""),
        "property_count": len(properties),
        "properties": formatted_props,
    }


@mcp.tool()
def generate_chart(
    ctx: Context,
    chart_type: Annotated[str, Field(description="Chart type: 'line', 'bar', 'pie', 'radar', 'scatter', etc.")],
    data: Annotated[dict[str, Any], Field(description="Data and configuration for the chart.")],
    title: Annotated[str | None, Field(description="Optional chart title.")] = None,
    include_tooltip: Annotated[bool, Field(description="Include tooltip config.", default=True)] = True,
    include_legend: Annotated[bool, Field(description="Include legend config.", default=True)] = True,
) -> dict[str, Any]:
    """
    Generate a validated ECharts option configuration.
    
    Takes chart type and data, returns a complete ECharts options dict ready for:
        myChart.setOption(option)
    
    Args:
        chart_type: Series type (line, bar, pie, radar, scatter, etc.)
        data: Chart data. Structure depends on chart_type:
            - line/bar: {"categories": [...], "values": [...]} or {"series": [...]}
            - pie: {"data": [{"name": "A", "value": 100}, ...]}
            - radar: {"indicator": [...], "data": [...]}
            - scatter: {"data": [[x,y], ...]}
        title: Optional title text
        include_tooltip: Add tooltip configuration
        include_legend: Add legend configuration
    
    Returns:
        Validated ECharts option dict.
    
    Example:
        generate_chart(
            chart_type="bar",
            data={"categories": ["Q1","Q2","Q3"], "values": [100, 200, 150]},
            title="Quarterly Sales"
        )
    """
    explorer = _get_explorer()
    
    # Validate chart type exists
    if not explorer.explore_series(chart_type):
        available = explorer.list_series_types()
        raise ToolError(f"Unknown chart type '{chart_type}'. Available: {', '.join(available[:15])}")
    
    # Build the option
    option: dict[str, Any] = {}
    
    # Title
    if title:
        option["title"] = {"text": title}
    
    # Tooltip
    if include_tooltip:
        if chart_type in ("pie", "radar", "funnel", "gauge"):
            option["tooltip"] = {"trigger": "item"}
        else:
            option["tooltip"] = {"trigger": "axis"}
    
    # Build series and axes based on chart type
    if chart_type in ("line", "bar", "scatter"):
        option.update(_build_cartesian_chart(chart_type, data, include_legend))
    elif chart_type == "pie":
        option.update(_build_pie_chart(data, include_legend))
    elif chart_type == "radar":
        option.update(_build_radar_chart(data, include_legend))
    else:
        # Generic: pass data directly to series
        option.update(_build_generic_chart(chart_type, data, include_legend))
    
    # Validate the generated option
    result = explorer.validate_option(option)
    
    if not result.valid:
        ctx.info(f"Validation errors: {result.errors}")
        raise ToolError(f"Generated config has errors: {result.errors}")
    
    if result.warnings:
        ctx.info(f"Validation warnings: {result.warnings}")
    
    return option


def _build_cartesian_chart(chart_type: str, data: dict[str, Any], include_legend: bool) -> dict[str, Any]:
    """Build line/bar/scatter chart config."""
    result: dict[str, Any] = {}
    
    # Handle different data formats
    if "categories" in data and "values" in data:
        # Simple format: {categories: [...], values: [...]}
        result["xAxis"] = {"type": "category", "data": data["categories"]}
        result["yAxis"] = {"type": "value"}
        result["series"] = [{"type": chart_type, "data": data["values"]}]
        
        if "name" in data:
            result["series"][0]["name"] = data["name"]
            if include_legend:
                result["legend"] = {"data": [data["name"]]}
    
    elif "series" in data:
        # Multiple series format
        categories = data.get("categories", data.get("xAxis", []))
        result["xAxis"] = {"type": "category", "data": categories}
        result["yAxis"] = {"type": "value"}
        
        series_list = data["series"]
        result["series"] = []
        legend_data = []
        
        for s in series_list:
            series_config = {"type": chart_type, "data": s.get("data", s.get("values", []))}
            if "name" in s:
                series_config["name"] = s["name"]
                legend_data.append(s["name"])
            # Pass through other properties
            for key in ("stack", "smooth", "areaStyle", "itemStyle", "label"):
                if key in s:
                    series_config[key] = s[key]
            result["series"].append(series_config)
        
        if include_legend and legend_data:
            result["legend"] = {"data": legend_data}
    
    elif "data" in data:
        # Scatter/direct data format
        result["xAxis"] = data.get("xAxis", {"type": "value"})
        result["yAxis"] = data.get("yAxis", {"type": "value"})
        result["series"] = [{"type": chart_type, "data": data["data"]}]
    
    else:
        # Pass through as-is
        result["xAxis"] = data.get("xAxis", {"type": "category"})
        result["yAxis"] = data.get("yAxis", {"type": "value"})
        result["series"] = [{"type": chart_type, **data}]
    
    return result


def _build_pie_chart(data: dict[str, Any], include_legend: bool) -> dict[str, Any]:
    """Build pie chart config."""
    result: dict[str, Any] = {}
    
    pie_data = data.get("data", data.get("values", []))
    
    series_config: dict[str, Any] = {
        "type": "pie",
        "data": pie_data,
    }
    
    # Optional properties
    if "radius" in data:
        series_config["radius"] = data["radius"]
    if "center" in data:
        series_config["center"] = data["center"]
    if "roseType" in data:
        series_config["roseType"] = data["roseType"]
    if "name" in data:
        series_config["name"] = data["name"]
    
    result["series"] = [series_config]
    
    # Legend from data names
    if include_legend:
        legend_data = [item.get("name") for item in pie_data if isinstance(item, dict) and "name" in item]
        if legend_data:
            result["legend"] = {"data": legend_data}
    
    return result


def _build_radar_chart(data: dict[str, Any], include_legend: bool) -> dict[str, Any]:
    """Build radar chart config."""
    result: dict[str, Any] = {}
    
    # Radar requires indicator
    indicator = data.get("indicator", [])
    result["radar"] = {"indicator": indicator}
    
    if "shape" in data:
        result["radar"]["shape"] = data["shape"]
    
    # Series data
    radar_data = data.get("data", data.get("series", []))
    series_config: dict[str, Any] = {
        "type": "radar",
        "data": radar_data,
    }
    if "name" in data:
        series_config["name"] = data["name"]
    
    result["series"] = [series_config]
    
    # Legend
    if include_legend:
        legend_data = [item.get("name") for item in radar_data if isinstance(item, dict) and "name" in item]
        if legend_data:
            result["legend"] = {"data": legend_data}
    
    return result


def _build_generic_chart(chart_type: str, data: dict[str, Any], include_legend: bool) -> dict[str, Any]:
    """Build generic chart config by passing data through."""
    result: dict[str, Any] = {}
    
    # Extract series data
    series_data = data.get("data", data.get("series", []))
    
    series_config: dict[str, Any] = {"type": chart_type}
    
    if isinstance(series_data, list):
        series_config["data"] = series_data
    else:
        series_config.update(data)
    
    # Pass through common properties
    for key in ("name", "radius", "center", "itemStyle", "label", "emphasis"):
        if key in data:
            series_config[key] = data[key]
    
    result["series"] = [series_config]
    
    # Add axes if needed for certain chart types
    if chart_type in ("heatmap", "boxplot", "candlestick"):
        result["xAxis"] = data.get("xAxis", {"type": "category"})
        result["yAxis"] = data.get("yAxis", {"type": "value"})
    
    if include_legend and "name" in data:
        result["legend"] = {"data": [data["name"]]}
    
    return result


if __name__ == "__main__":
    mcp.run()
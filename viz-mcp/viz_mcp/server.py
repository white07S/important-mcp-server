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
        explorer.list_series_types()
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
        "ECharts visualization server. Use resources to explore chart schemas. "
        "Use generate_chart to create validated ECharts option configs."
    ),
    lifespan=lifespan,
)


# =============================================================================
# RESOURCES
# =============================================================================

@mcp.resource("echarts://charts")
def resource_list_charts() -> dict[str, Any]:
    """List all available ECharts chart types and components."""
    explorer = _get_explorer()
    return {
        "chart_types": explorer.list_series_types(),
        "components": explorer.list_components(),
    }


@mcp.resource("echarts://charts/{chart_type}")
def resource_chart_schema(chart_type: str) -> dict[str, Any]:
    """Get schema for a specific chart type."""
    explorer = _get_explorer()
    series = explorer.explore_series(chart_type)
    if not series:
        return {"error": f"Unknown chart type: {chart_type}", "available": explorer.list_series_types()}
    
    properties = series.get("properties", {})
    formatted = {}
    for prop, info in properties.items():
        formatted[prop] = {
            "type": info.get("type"),
            "default": info.get("default"),
            "description": info.get("description", "")[:200],
        }
        if info.get("options"):
            formatted[prop]["options"] = info["options"]
    
    return {
        "chart_type": chart_type,
        "description": series.get("description", ""),
        "properties": formatted,
    }


@mcp.resource("echarts://components/{component}")
def resource_component_schema(component: str) -> dict[str, Any]:
    """Get schema for a component (xAxis, yAxis, tooltip, legend, etc.)."""
    explorer = _get_explorer()
    comp = explorer.explore_component(component)
    if not comp:
        return {"error": f"Unknown component: {component}", "available": explorer.list_components()}
    
    properties = comp.get("properties", {})
    formatted = {}
    for prop, info in properties.items():
        formatted[prop] = {
            "type": info.get("type"),
            "default": info.get("default"),
            "description": info.get("description", "")[:150],
        }
    
    return {
        "component": component,
        "description": comp.get("description", ""),
        "properties": formatted,
    }


@mcp.resource("echarts://search/{keyword}")
def resource_search(keyword: str) -> dict[str, Any]:
    """Search schema for properties matching keyword."""
    explorer = _get_explorer()
    results = explorer.search(keyword, limit=20)
    return {
        "query": keyword,
        "matches": [{"kind": kind, "path": path} for kind, path in results],
    }


# =============================================================================
# TOOLS
# =============================================================================

@mcp.tool()
def ping() -> str:
    """Health check."""
    return "pong"


@mcp.tool()
def list_chart_types() -> dict[str, Any]:
    """List all available ECharts chart types and components."""
    explorer = _get_explorer()
    return {
        "chart_types": explorer.list_series_types(),
        "components": explorer.list_components(),
    }


@mcp.tool()
def describe_chart(
    chart_type: Annotated[str, Field(description="Chart type: 'line', 'bar', 'pie', 'radar', etc.")],
) -> dict[str, Any]:
    """
    Get full schema for a chart type.
    
    Use this to discover available properties before calling generate_chart.
    Pass relevant properties via chart_spec parameter in generate_chart.
    """
    explorer = _get_explorer()
    series = explorer.explore_series(chart_type)
    if not series:
        raise ToolError(f"Unknown chart type '{chart_type}'. Use list_chart_types first.")
    
    properties = series.get("properties", {})
    formatted = {}
    for prop, info in properties.items():
        formatted[prop] = {
            "type": info.get("type"),
            "default": info.get("default"),
            "description": info.get("description", "")[:300],
        }
        if info.get("options"):
            formatted[prop]["options"] = info["options"]
    
    return {
        "chart_type": chart_type,
        "description": series.get("description", ""),
        "property_count": len(properties),
        "properties": formatted,
    }


@mcp.tool()
def generate_chart(
    ctx: Context,
    chart_type: Annotated[str, Field(description="Chart type: 'line', 'bar', 'pie', 'radar', 'scatter', etc.")],
    data: Annotated[Any, Field(description="Chart data. Format depends on chart type.")],
    chart_spec: Annotated[
        dict[str, Any] | None,
        Field(description="Series properties from describe_chart (smooth, stack, radius, itemStyle, etc.).")
    ] = None,
    title: Annotated[str | None, Field(description="Chart title.")] = None,
    tooltip: Annotated[dict[str, Any] | None, Field(description="Tooltip config. {} for defaults, None to omit.")] = None,
    legend: Annotated[dict[str, Any] | None, Field(description="Legend config. {} for defaults, None to omit.")] = None,
    x_axis: Annotated[dict[str, Any] | None, Field(description="xAxis config for cartesian charts.")] = None,
    y_axis: Annotated[dict[str, Any] | None, Field(description="yAxis config for cartesian charts.")] = None,
    grid: Annotated[dict[str, Any] | None, Field(description="Grid config.")] = None,
    extra: Annotated[dict[str, Any] | None, Field(description="Additional top-level properties (radar, polar, geo, visualMap, etc.).")] = None,
) -> dict[str, Any]:
    """
    Generate a validated ECharts option configuration.
    
    Workflow:
        1. Call describe_chart(chart_type) to see available properties
        2. Call generate_chart with data and chart_spec containing desired properties
    
    Args:
        chart_type: Series type (line, bar, pie, radar, scatter, etc.)
        data: Chart data - structure varies by chart type
        chart_spec: Series-specific properties from describe_chart
        title: Optional title text
        tooltip: Tooltip config, {} for auto, None to omit
        legend: Legend config, {} for auto, None to omit  
        x_axis: xAxis config for cartesian charts
        y_axis: yAxis config for cartesian charts
        grid: Grid config
        extra: Top-level properties (radar, polar, geo, etc.)
    
    Returns:
        Validated ECharts option dict ready for setOption()
    
    Examples:
        # Simple bar
        generate_chart("bar", [10, 20, 30], x_axis={"data": ["A", "B", "C"]})
        
        # Smooth line with area
        generate_chart("line", [10, 20, 30], 
            chart_spec={"smooth": True, "areaStyle": {}},
            x_axis={"data": ["A", "B", "C"]})
        
        # Donut pie
        generate_chart("pie", 
            [{"name": "A", "value": 100}, {"name": "B", "value": 200}],
            chart_spec={"radius": ["40%", "70%"]})
        
        # Radar
        generate_chart("radar",
            [{"name": "Budget", "value": [80, 90, 70]}],
            extra={"radar": {"indicator": [
                {"name": "Sales", "max": 100},
                {"name": "Tech", "max": 100},
                {"name": "Support", "max": 100}
            ]}})
    """
    explorer = _get_explorer()
    
    # Validate chart type
    if not explorer.explore_series(chart_type):
        available = explorer.list_series_types()
        raise ToolError(f"Unknown chart type '{chart_type}'. Available: {', '.join(available[:15])}")
    
    option: dict[str, Any] = {}
    
    # Title
    if title:
        option["title"] = {"text": title}
    
    # Grid
    if grid is not None:
        option["grid"] = grid
    
    # Axes
    if x_axis is not None:
        option["xAxis"] = {"type": "category", **x_axis}
    if y_axis is not None:
        option["yAxis"] = {"type": "value", **y_axis}
    
    # Auto-add axes for cartesian charts
    cartesian_types = {"line", "bar", "scatter", "effectScatter", "candlestick", "boxplot", "heatmap"}
    if chart_type in cartesian_types:
        if "xAxis" not in option:
            option["xAxis"] = {"type": "category"}
        if "yAxis" not in option:
            option["yAxis"] = {"type": "value"}
    
    # Tooltip
    if tooltip is not None:
        if tooltip == {}:
            if chart_type in {"pie", "radar", "funnel", "gauge", "treemap", "sunburst"}:
                option["tooltip"] = {"trigger": "item"}
            else:
                option["tooltip"] = {"trigger": "axis"}
        else:
            option["tooltip"] = tooltip
    
    # Extra top-level properties first (radar, polar, etc.)
    if extra:
        for key, value in extra.items():
            option[key] = value
    
    # Build series
    series_config: dict[str, Any] = {"type": chart_type, "data": data}
    
    # Merge chart_spec
    if chart_spec:
        series_config.update(chart_spec)
    
    option["series"] = [series_config]
    
    # Legend
    if legend is not None:
        if legend == {}:
            legend_data = _extract_legend_data(data)
            option["legend"] = {"data": legend_data} if legend_data else {}
        else:
            option["legend"] = legend
    
    # Validate
    result = explorer.validate_option(option)
    
    if not result.valid:
        ctx.info(f"Validation errors: {result.errors}")
        raise ToolError(f"Invalid config: {result.errors}")
    
    if result.warnings:
        ctx.info(f"Warnings: {result.warnings}")
    
    return option


def _extract_legend_data(data: Any) -> list[str]:
    """Extract legend names from data."""
    if isinstance(data, list):
        return [item["name"] for item in data if isinstance(item, dict) and "name" in item]
    return []


if __name__ == "__main__":
    mcp.run()
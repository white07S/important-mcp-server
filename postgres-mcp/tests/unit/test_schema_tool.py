import json

import pytest

from postgres_mcp.server import explore_schema


@pytest.mark.asyncio
async def test_explore_schema_returns_all_tables_from_file():
    response = await explore_schema()
    assert len(response) == 1

    payload = json.loads(response[0].text)
    assert payload["source"].endswith("schema/database_schema.json")
    assert "tables" in payload
    assert len(payload["tables"]) > 0


@pytest.mark.asyncio
async def test_explore_schema_filters_and_summarizes_table():
    response = await explore_schema(table_name="employees", detail_level="summary", include_relationships=False)
    assert len(response) == 1

    payload = json.loads(response[0].text)
    assert len(payload["tables"]) == 1
    table = payload["tables"][0]
    assert table["table_name"] == "employees"
    assert "foreign_keys" not in table
    assert any(column["name"] == "emp_no" for column in table["columns"])

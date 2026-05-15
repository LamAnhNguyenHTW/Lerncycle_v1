"""Inspect the current Neo4j schema used by GraphRAG and learning graphs."""

from __future__ import annotations

import argparse
import json
from typing import Any

from rag_pipeline.config import WorkerConfig


def inspect_schema(driver: Any, database: str | None = None) -> dict[str, Any]:
    """Collect labels, relationship types, schema objects, and label properties."""
    labels = _single_column(driver, "CALL db.labels()", "label", database)
    relationship_types = _single_column(
        driver,
        "CALL db.relationshipTypes()",
        "relationshipType",
        database,
    )
    constraints = _records(driver, "SHOW CONSTRAINTS", database=database)
    indexes = _records(driver, "SHOW INDEXES", database=database)
    property_keys_by_label = {
        label: _property_keys_for_label(driver, label, database)
        for label in labels
    }
    return {
        "labels": labels,
        "relationship_types": relationship_types,
        "constraints": constraints,
        "indexes": indexes,
        "property_keys_by_label": property_keys_by_label,
    }


def render_markdown(inventory: dict[str, Any]) -> str:
    """Render an inventory dictionary as concise Markdown."""
    lines = ["# Neo4j Schema Inventory", ""]
    lines.extend(_list_section("Labels", inventory.get("labels", [])))
    lines.extend(_list_section("Relationship Types", inventory.get("relationship_types", [])))
    lines.extend(_json_section("Constraints", inventory.get("constraints", [])))
    lines.extend(_json_section("Indexes", inventory.get("indexes", [])))
    lines.append("## Property Keys by Label")
    for label, keys in sorted(inventory.get("property_keys_by_label", {}).items()):
        rendered = ", ".join(f"`{key}`" for key in keys) if keys else "_No sampled properties_"
        lines.append(f"- `{label}`: {rendered}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point using the configured Neo4j connection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    config = WorkerConfig.from_env()
    if not config.neo4j_uri or not config.neo4j_user or not config.neo4j_password:
        raise RuntimeError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are required")

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password),
    )
    try:
        inventory = inspect_schema(driver, database=config.neo4j_database)
    finally:
        driver.close()

    if args.format == "json":
        print(json.dumps(inventory, indent=2, sort_keys=True))
    else:
        print(render_markdown(inventory))


def _single_column(
    driver: Any,
    statement: str,
    key: str,
    database: str | None,
) -> list[str]:
    return [str(record[key]) for record in _records(driver, statement, database=database)]


def _property_keys_for_label(
    driver: Any,
    label: str,
    database: str | None,
) -> list[str]:
    statement = f"MATCH (n:`{_escape_label(label)}`) RETURN keys(n) AS keys LIMIT 1"
    records = _records(driver, statement, database=database)
    if not records:
        return []
    keys = records[0].get("keys") or []
    return sorted(str(key) for key in keys)


def _records(
    driver: Any,
    statement: str,
    parameters: dict[str, Any] | None = None,
    database: str | None = None,
) -> list[dict[str, Any]]:
    session = driver.session(database=database)
    if hasattr(session, "__enter__"):
        with session as active_session:
            return _materialize_records(active_session.run(statement, parameters or {}))
    try:
        return _materialize_records(session.run(statement, parameters or {}))
    finally:
        close = getattr(session, "close", None)
        if close:
            close()


def _materialize_records(result: Any) -> list[dict[str, Any]]:
    records = []
    for record in result or []:
        if hasattr(record, "data"):
            records.append(record.data())
        elif isinstance(record, dict):
            records.append(dict(record))
    return records


def _escape_label(label: str) -> str:
    return label.replace("`", "``")


def _list_section(title: str, values: list[str]) -> list[str]:
    lines = [f"## {title}"]
    lines.extend(f"- `{value}`" for value in values)
    if not values:
        lines.append("- _None found_")
    lines.append("")
    return lines


def _json_section(title: str, values: list[dict[str, Any]]) -> list[str]:
    return [
        f"## {title}",
        "```json",
        json.dumps(_json_safe(values), indent=2, sort_keys=True),
        "```",
        "",
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


if __name__ == "__main__":
    main()

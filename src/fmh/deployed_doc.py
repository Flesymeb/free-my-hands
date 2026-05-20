from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from fmh.config import ReusableWorkersConfig


class DocxClient(Protocol):
    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        ...

    def _patch(self, path: str, *, json: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class DeployedDocUpdateResult:
    table_block_id: str
    row_number: int
    updated_columns: list[str]


def update_deployed_models_row(
    client: DocxClient,
    config: ReusableWorkersConfig,
    *,
    row_index: int,
    values: dict[str, str],
) -> DeployedDocUpdateResult:
    doc_token = config.deployed_models_doc_token
    table = _find_deployed_models_table(client, doc_token)
    table_block_id = table["block_id"]
    columns = int(table["column_size"])
    cells = table["cells"]
    rows = int(table["row_size"])
    if row_index <= 0 or row_index >= rows:
        raise ValueError(f"row_index out of range: {row_index}")

    headers = table["headers"]
    updates: list[str] = []
    for column_name, value in values.items():
        normalized = _normalize_header(column_name)
        if normalized not in headers:
            continue
        column = headers[normalized]
        cell_id = cells[row_index * columns + column]
        text_block_id = _first_text_child(table["blocks"], cell_id)
        _patch_text_block(client, doc_token, text_block_id, str(value))
        updates.append(column_name)
    return DeployedDocUpdateResult(table_block_id=table_block_id, row_number=row_index, updated_columns=updates)


def _find_deployed_models_table(client: DocxClient, doc_token: str) -> dict[str, Any]:
    root = client._get(f"/docx/v1/documents/{doc_token}/blocks/{doc_token}")
    children = root.get("data", {}).get("block", {}).get("children", [])
    for block_id in children:
        block = client._get(f"/docx/v1/documents/{doc_token}/blocks/{block_id}").get("data", {}).get("block", {})
        if int(block.get("block_type") or 0) != 31:
            continue
        table = block.get("table") if isinstance(block.get("table"), dict) else {}
        prop = table.get("property") if isinstance(table.get("property"), dict) else {}
        column_size = int(prop.get("column_size") or 0)
        row_size = int(prop.get("row_size") or 0)
        cells = table.get("cells") if isinstance(table.get("cells"), list) else []
        if column_size <= 0 or row_size <= 0 or not cells:
            continue
        tree = client._get(
            f"/docx/v1/documents/{doc_token}/blocks/{block_id}/children",
            params={"page_size": 500, "document_revision_id": -1, "with_descendants": "true"},
        )
        items = tree.get("data", {}).get("items", [])
        blocks = {str(item.get("block_id")): item for item in items if item.get("block_id")}
        headers = {
            _normalize_header(_cell_text(blocks, cells[column])): column
            for column in range(column_size)
        }
        required = {"模型", "模型id", "地址"}
        if required.issubset(headers):
            return {
                "block_id": block_id,
                "column_size": column_size,
                "row_size": row_size,
                "cells": cells,
                "headers": headers,
                "blocks": blocks,
            }
    raise ValueError("deployed models table not found")


def _first_text_child(blocks: dict[str, dict[str, Any]], cell_id: str) -> str:
    cell = blocks.get(cell_id) or {}
    for child_id in cell.get("children", []):
        child = blocks.get(str(child_id)) or {}
        if int(child.get("block_type") or 0) == 2:
            return str(child_id)
    raise ValueError(f"table cell has no text child: {cell_id}")


def _cell_text(blocks: dict[str, dict[str, Any]], cell_id: str) -> str:
    cell = blocks.get(cell_id) or {}
    parts: list[str] = []
    for child_id in cell.get("children", []):
        child = blocks.get(str(child_id)) or {}
        text = child.get("text") if isinstance(child.get("text"), dict) else {}
        for element in text.get("elements", []):
            if not isinstance(element, dict):
                continue
            text_run = element.get("text_run") if isinstance(element.get("text_run"), dict) else {}
            parts.append(str(text_run.get("content") or ""))
    return "".join(parts).strip()


def _patch_text_block(client: DocxClient, doc_token: str, block_id: str, content: str) -> None:
    client._patch(
        f"/docx/v1/documents/{doc_token}/blocks/{block_id}",
        json={
            "update_text_elements": {
                "elements": [
                    {
                        "text_run": {
                            "content": content,
                        }
                    }
                ]
            }
        },
    )


def _normalize_header(value: str) -> str:
    return "".join(str(value).split())

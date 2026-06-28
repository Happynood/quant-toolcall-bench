from __future__ import annotations

import json
from typing import Any

from quantcall.datasets.base import NormalizedInstance, ToolCall, ToolSpec


def _parse_json_field(field: Any) -> Any:
    """Parse a field that may be a JSON string or already a Python object."""
    if isinstance(field, str):
        try:
            return json.loads(field)
        except json.JSONDecodeError:
            return []
    return field


def normalize_xlam_instance(raw: dict[str, Any]) -> NormalizedInstance:
    inst_id = str(raw.get("id", raw.get("question_id", "unknown")))
    query = raw.get("query", raw.get("question", ""))

    raw_tools = _parse_json_field(raw.get("tools", []))
    if not isinstance(raw_tools, list):
        raw_tools = []

    tools: list[ToolSpec] = []
    for t in raw_tools:
        if not isinstance(t, dict):
            continue
        params = t.get("parameters", t.get("arguments", {}))
        schema: dict[str, Any] = {
            "type": "object",
            "properties": dict(params.get("properties", {})),
        }
        req = params.get("required", [])
        if req:
            schema["required"] = req
        tools.append(
            ToolSpec(
                name=t["name"],
                description=t.get("description", ""),
                json_schema=schema,
            )
        )

    raw_answers = _parse_json_field(raw.get("answers", []))
    if not isinstance(raw_answers, list):
        raw_answers = []

    gt_calls: list[ToolCall] = []
    for ans in raw_answers:
        if not isinstance(ans, dict):
            continue
        name = ans.get("name", "")
        args = ans.get("arguments", ans.get("args", {}))
        if name:
            gt_calls.append(ToolCall(name=name, arguments=args if isinstance(args, dict) else {}))

    return NormalizedInstance(
        id=f"T4-xlam-{inst_id}",
        tier="T4",
        category="xlam",
        query=query,
        tools=tools,
        ground_truth_calls=gt_calls,
        expects_call=len(gt_calls) > 0,
    )

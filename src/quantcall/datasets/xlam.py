from __future__ import annotations

import json
from typing import Any

from quantcall.datasets.base import NormalizedInstance, ToolCall, ToolSpec

XLAM_UNGATED_REPO = "minpeter/xlam-function-calling-60k-parsed"
XLAM_GATED_REPO = "Salesforce/xlam-function-calling-60k"


def _parse_json_field(field: Any) -> Any:
    if isinstance(field, str):
        try:
            return json.loads(field)
        except json.JSONDecodeError:
            return []
    return field


def _parse_tool_spec(t: Any) -> ToolSpec | None:
    """Parse one tool definition — handles flat (Salesforce) and OpenAI-wrapped formats."""
    if not isinstance(t, dict):
        return None
    # OpenAI format: {"type": "function", "function": {"name": ..., "parameters": ...}}
    if "function" in t and isinstance(t["function"], dict):
        fn = t["function"]
        name = fn.get("name", "")
        description = fn.get("description", "")
        params: dict[str, Any] = fn.get("parameters", {})
    else:
        # Flat format: {"name": ..., "description": ..., "parameters": ...}
        name = t.get("name", "")
        description = t.get("description", "")
        params = t.get("parameters", t.get("arguments", {}))

    if not name:
        return None

    schema: dict[str, Any] = {
        "type": "object",
        "properties": dict(params.get("properties", {})),
    }
    req = params.get("required", [])
    if req:
        schema["required"] = req

    return ToolSpec(name=name, description=description, json_schema=schema)


def _parse_gt_calls(raw_answers: Any) -> list[ToolCall]:
    answers = _parse_json_field(raw_answers)
    if not isinstance(answers, list):
        return []
    calls: list[ToolCall] = []
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        name = ans.get("name", "")
        args = ans.get("arguments", ans.get("args", {}))
        if name:
            calls.append(ToolCall(name=name, arguments=args if isinstance(args, dict) else {}))
    return calls


def normalize_xlam_instance(raw: dict[str, Any]) -> NormalizedInstance:
    """Normalize a Salesforce xLAM instance (gated dataset, used with --use-gated-xlam)."""
    inst_id = str(raw.get("id", raw.get("question_id", "unknown")))
    query = raw.get("query", raw.get("question", ""))

    raw_tools = _parse_json_field(raw.get("tools", []))
    if not isinstance(raw_tools, list):
        raw_tools = []

    tools = [s for t in raw_tools if (s := _parse_tool_spec(t)) is not None]
    gt_calls = _parse_gt_calls(raw.get("answers", []))

    return NormalizedInstance(
        id=f"T4-xlam-{inst_id}",
        tier="T4",
        category="xlam",
        query=query,
        tools=tools,
        ground_truth_calls=gt_calls,
        expects_call=len(gt_calls) > 0,
    )


def normalize_xlam_parsed_instance(raw: dict[str, Any]) -> NormalizedInstance:
    """Normalize a minpeter/xlam-function-calling-60k-parsed instance (ungated default).

    Format: {messages: [{role, content}], tools: [...], extra: {answers: ...}}
    """
    inst_id = str(raw.get("id", "unknown"))

    # Last user message is the query
    query = ""
    for msg in reversed(raw.get("messages", [])):
        if isinstance(msg, dict) and msg.get("role") == "user":
            query = msg.get("content", "")
            break

    raw_tools = raw.get("tools", [])
    if not isinstance(raw_tools, list):
        raw_tools = []
    tools = [s for t in raw_tools if (s := _parse_tool_spec(t)) is not None]

    extra = raw.get("extra", {})
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = {}
    raw_answers = extra.get("answers", []) if isinstance(extra, dict) else []
    gt_calls = _parse_gt_calls(raw_answers)

    return NormalizedInstance(
        id=f"T4-xlam-{inst_id}",
        tier="T4",
        category="xlam",
        query=query,
        tools=tools,
        ground_truth_calls=gt_calls,
        expects_call=len(gt_calls) > 0,
    )

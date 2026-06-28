from __future__ import annotations

from typing import Any

from quantcall.datasets.base import NormalizedInstance, ToolCall, ToolSpec


def _extract_schema(params: dict[str, Any]) -> dict[str, Any]:
    if not params:
        return {"type": "object", "properties": {}}
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    for k, v in params.get("properties", {}).items():
        schema["properties"][k] = dict(v)
    required = params.get("required", [])
    if required:
        schema["required"] = required
    return schema


def normalize_toolace_instance(raw: dict[str, Any]) -> NormalizedInstance:
    inst_id = str(raw.get("id", "unknown"))
    query = raw.get("query", raw.get("instruction", ""))

    raw_tools = raw.get("tools", [])
    tools: list[ToolSpec] = []
    for t in raw_tools:
        tools.append(
            ToolSpec(
                name=t["name"],
                description=t.get("description", ""),
                json_schema=_extract_schema(t.get("parameters", {})),
            )
        )

    answers = raw.get("answers", [])
    gt_calls: list[ToolCall] = []
    for ans in answers:
        if isinstance(ans, dict):
            name = ans.get("name", "")
            args = ans.get("arguments", ans.get("args", {}))
            if name:
                safe_args = args if isinstance(args, dict) else {}
                gt_calls.append(ToolCall(name=name, arguments=safe_args))

    return NormalizedInstance(
        id=f"T3-toolace-{inst_id}",
        tier="T3",
        category="toolace",
        query=query,
        tools=tools,
        ground_truth_calls=gt_calls,
        expects_call=len(gt_calls) > 0,
    )

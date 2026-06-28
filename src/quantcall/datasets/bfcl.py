from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quantcall.datasets.base import NormalizedInstance, ToolCall, ToolSpec

_TIER_MAP: dict[str, str] = {
    "simple": "T1",
    "multiple": "T1",
    "parallel": "T2",
    "parallel_multiple": "T2",
    "irrelevance": "T6",
    "relevance": "T6",
}

_DEFAULT_CATEGORY_FILES: dict[str, str] = {
    "simple": "gorilla_openfunctions_v1_test_simple.json",
    "multiple": "gorilla_openfunctions_v1_test_multiple_function.json",
    "parallel": "gorilla_openfunctions_v1_test_parallel_function.json",
    "parallel_multiple": "gorilla_openfunctions_v1_test_parallel_multiple_function.json",
    "irrelevance": "gorilla_openfunctions_v1_test_irrelevance.json",
}


def _extract_json_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    """Extract or reconstruct a JSON Schema from a BFCL parameter spec."""
    if not parameters:
        return {"type": "object", "properties": {}}
    props = {}
    required: list[str] = parameters.get("required", [])
    for name, spec in parameters.get("properties", {}).items():
        props[name] = {k: v for k, v in spec.items()}
    schema: dict[str, Any] = {
        "type": "object",
        "properties": props,
    }
    if required:
        schema["required"] = required
    return schema


def normalize_bfcl_instance(
    raw: dict[str, Any],
    category: str,
) -> NormalizedInstance:
    tier = _TIER_MAP.get(category, "T1")
    question_id = str(raw.get("question_id", raw.get("id", "unknown")))
    query = raw.get("question", raw.get("prompt", ""))

    functions = raw.get("function", raw.get("functions", []))
    if isinstance(functions, dict):
        functions = [functions]

    tools: list[ToolSpec] = []
    for fn in functions:
        params = fn.get("parameters", {})
        tools.append(
            ToolSpec(
                name=fn["name"],
                description=fn.get("description", ""),
                json_schema=_extract_json_schema(params),
            )
        )

    gt_calls: list[ToolCall] = []
    expects_call = category not in ("irrelevance", "relevance")

    ground_truth = raw.get("ground_truth", raw.get("answers", []))
    if isinstance(ground_truth, dict):
        ground_truth = [ground_truth]

    for gt_item in ground_truth or []:
        if isinstance(gt_item, dict):
            for fn_name, args in gt_item.items():
                if isinstance(args, dict):
                    gt_calls.append(ToolCall(name=fn_name, arguments=args))
                    break

    if not gt_calls:
        expects_call = False

    return NormalizedInstance(
        id=f"{tier}-bfcl-{question_id}",
        tier=tier,
        category=category,
        query=query,
        tools=tools,
        ground_truth_calls=gt_calls,
        expects_call=expects_call,
    )


def load_bfcl(
    categories: list[str] | None = None,
    data_dir: str | Path | None = None,
    category_files: dict[str, str] | None = None,
) -> list[NormalizedInstance]:
    """Load BFCL instances from JSONL/JSON files (manual reader, never load_dataset).

    Parameters
    ----------
    categories: list of category names to load (e.g. ["simple", "multiple"])
    data_dir: directory containing BFCL JSON files (optional if category_files given)
    category_files: explicit mapping of category → file path (overrides data_dir)
    """
    if categories is None:
        categories = ["simple", "multiple"]

    cat_files: dict[str, str] = {}
    if category_files:
        cat_files = category_files
    elif data_dir is not None:
        base = Path(data_dir)
        for cat in categories:
            default_name = _DEFAULT_CATEGORY_FILES.get(cat, f"{cat}.json")
            cat_files[cat] = str(base / default_name)
    else:
        raise ValueError("Provide either data_dir or category_files")

    instances: list[NormalizedInstance] = []
    for cat in categories:
        path = cat_files.get(cat)
        if path is None:
            continue
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"BFCL file not found: {p}")

        with open(p) as f:
            text = f.read().strip()

        if not text:
            continue

        if text.startswith("["):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]

        for rec in records:
            instances.append(normalize_bfcl_instance(rec, category=cat))

    return instances

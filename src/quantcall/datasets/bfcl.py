from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quantcall.datasets.base import NormalizedInstance, ToolCall, ToolSpec

# Map BFCL v4 category names → tier codes
_TIER_MAP: dict[str, str] = {
    "simple_python": "T1",
    "simple": "T1",
    "multiple": "T1",
    "parallel": "T2",
    "parallel_multiple": "T2",
    "irrelevance": "T6",
    "relevance": "T6",
    # legacy v1 names
    "simple_function": "T1",
    "multiple_function": "T1",
}

# Default BFCL v4 question files per category
_DEFAULT_QUESTION_FILES: dict[str, str] = {
    "simple_python": "BFCL_v4_simple_python.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel": "BFCL_v4_parallel.json",
    "parallel_multiple": "BFCL_v4_parallel_multiple.json",
    "irrelevance": "BFCL_v4_irrelevance.json",
    # legacy v1 names (still supported if files are present)
    "simple": "gorilla_openfunctions_v1_test_simple.json",
    "multiple_function": "gorilla_openfunctions_v1_test_multiple_function.json",
}

# Default BFCL v4 possible-answer files (None = abstention tier, no answers)
_DEFAULT_ANSWER_FILES: dict[str, str | None] = {
    "simple_python": "possible_answer/BFCL_v4_simple_python.json",
    "multiple": "possible_answer/BFCL_v4_multiple.json",
    "parallel": "possible_answer/BFCL_v4_parallel.json",
    "parallel_multiple": "possible_answer/BFCL_v4_parallel_multiple.json",
    "irrelevance": None,  # abstention: no call expected
    "simple": None,  # v1 legacy: ground truth embedded in question file
    "multiple_function": None,
}


def _extract_json_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    """Convert BFCL parameter spec to JSON Schema."""
    if not parameters:
        return {"type": "object", "properties": {}}
    props: dict[str, Any] = {}
    for name, spec in parameters.get("properties", {}).items():
        entry: dict[str, Any] = {}
        for key in ("type", "description", "enum", "default"):
            if key in spec:
                entry[key] = spec[key]
        # Normalise BFCL type aliases
        if entry.get("type") == "dict":
            entry["type"] = "object"
        if entry.get("type") == "list":
            entry["type"] = "array"
        props[name] = entry
    schema: dict[str, Any] = {"type": "object", "properties": props}
    required: list[str] = parameters.get("required", [])
    if required:
        schema["required"] = required
    return schema


def _first_value(val: Any) -> Any:
    """Resolve a BFCL v4 possible-answer value to a single canonical value.

    v4 stores each argument's valid values as a list, e.g. `{"unit": ["units", ""]}`.
    We pick the first non-empty value as the canonical ground-truth.
    AC will be conservative (may undercount correct calls with alternative values),
    but the degradation signal across quantisation levels remains valid.
    """
    if isinstance(val, list):
        for v in val:
            if v != "" and v is not None:
                return v
        return val[0] if val else ""
    return val


def _parse_v4_ground_truth(
    gt_list: list[dict[str, Any]],
) -> list[ToolCall]:
    """Parse BFCL v4 ground_truth into ToolCall list."""
    calls: list[ToolCall] = []
    for gt_item in gt_list:
        for fn_name, args_spec in gt_item.items():
            if not isinstance(args_spec, dict):
                continue
            args = {k: _first_value(v) for k, v in args_spec.items()}
            calls.append(ToolCall(name=fn_name, arguments=args))
    return calls


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text().strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)  # JSON array (v1 style)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_bfcl_instance(
    raw: dict[str, Any],
    category: str,
    ground_truth_lookup: dict[str, list[ToolCall]] | None = None,
) -> NormalizedInstance:
    """Convert a raw BFCL v4 (or v1) record to NormalizedInstance.

    Parameters
    ----------
    raw:
        One JSONL record from a BFCL question file.
    category:
        Category string, e.g. "simple_python", "multiple", "irrelevance".
    ground_truth_lookup:
        Pre-loaded mapping from record id → list[ToolCall], built from the
        corresponding possible_answer file.  If None, falls back to the legacy
        ``ground_truth`` / ``answers`` key embedded in ``raw``.
    """
    tier = _TIER_MAP.get(category, "T1")
    record_id = str(raw.get("id", raw.get("question_id", "unknown")))

    # Extract query from v4 format (nested conversation list) or v1 flat string
    question_field = raw.get("question", raw.get("prompt", ""))
    if isinstance(question_field, list):
        # v4: [[{"role": "user", "content": "..."}], ...]
        first_turn = question_field[0] if question_field else []
        if first_turn and isinstance(first_turn, list):
            query = first_turn[0].get("content", "") if first_turn else ""
        elif first_turn and isinstance(first_turn, dict):
            query = first_turn.get("content", "")
        else:
            query = ""
    else:
        query = str(question_field)

    # Parse tool specs
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

    # Resolve ground truth
    expects_call = category not in ("irrelevance", "relevance")
    gt_calls: list[ToolCall] = []

    if ground_truth_lookup is not None:
        gt_calls = ground_truth_lookup.get(record_id, [])
    else:
        # Legacy v1: ground truth embedded in record
        ground_truth = raw.get("ground_truth", raw.get("answers", []))
        if isinstance(ground_truth, dict):
            ground_truth = [ground_truth]
        for gt_item in ground_truth or []:
            if isinstance(gt_item, dict):
                for fn_name, args in gt_item.items():
                    if isinstance(args, dict):
                        gt_calls.append(ToolCall(name=fn_name, arguments=args))
                        break

    if not gt_calls and expects_call:
        expects_call = False

    return NormalizedInstance(
        id=f"{tier}-bfcl-{record_id}",
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
    answer_files: dict[str, str | None] | None = None,
) -> list[NormalizedInstance]:
    """Load BFCL instances from local files (never uses datasets.load_dataset).

    Parameters
    ----------
    categories:
        Category names to load, e.g. ``["simple_python", "multiple", "irrelevance"]``.
        Defaults to ``["simple_python", "multiple"]``.
    data_dir:
        Directory containing BFCL JSONL files.  Required unless ``category_files``
        is supplied.
    category_files:
        Explicit per-category mapping to question file paths (overrides data_dir).
    answer_files:
        Explicit per-category mapping to possible_answer file paths, or ``None``
        for abstention tiers.  When not supplied, inferred from data_dir.
    """
    if categories is None:
        categories = ["simple_python", "multiple"]

    if data_dir is None and category_files is None:
        raise ValueError("Provide either data_dir or category_files")

    base = Path(data_dir) if data_dir else None

    instances: list[NormalizedInstance] = []
    for cat in categories:
        # Resolve question file path
        if category_files and cat in category_files:
            q_path = Path(category_files[cat])
        elif base is not None:
            default_name = _DEFAULT_QUESTION_FILES.get(cat, f"{cat}.json")
            q_path = base / default_name
        else:
            continue

        if not q_path.exists():
            raise FileNotFoundError(f"BFCL question file not found: {q_path}")

        records = _load_jsonl(q_path)

        # Resolve possible-answer file path
        gt_lookup: dict[str, list[ToolCall]] | None = None
        if answer_files is not None:
            ans_path_str = answer_files.get(cat)
        else:
            ans_rel = _DEFAULT_ANSWER_FILES.get(cat)
            ans_path_str = str(base / ans_rel) if (base and ans_rel) else None

        if ans_path_str:
            ans_path = Path(ans_path_str)
            if ans_path.exists():
                ans_records = _load_jsonl(ans_path)
                gt_lookup = {}
                for ans in ans_records:
                    rec_id = str(ans.get("id", ""))
                    gt_lookup[rec_id] = _parse_v4_ground_truth(ans.get("ground_truth", []))

        for rec in records:
            instances.append(
                normalize_bfcl_instance(rec, category=cat, ground_truth_lookup=gt_lookup)
            )

    return instances

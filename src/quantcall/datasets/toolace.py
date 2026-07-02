"""ToolACE (Team-ACE/ToolACE, CC-BY-NC 4.0) adapter.

Real schema (verified against the live HF dataset, NOT assumed): each row is
``{"system": str, "conversations": [{"from": "user"|"assistant"|"tool", "value": str}, ...]}``.
The tool list is a JSON array embedded in prose inside ``system`` (not a
separate structured field), and ground-truth calls in ``assistant`` turns are
written as a Python-call-list string, e.g.
``[SEC Filings(identifier="AAPL"), Some Other Tool(x="y")]`` -- note function
names may contain spaces, unlike BFCL's dotted-identifier names.

QuantCall's NormalizedInstance is single-turn (one query -> one expected
call), but ToolACE conversations are multi-turn. Conservative, explicit
choice: only the FIRST (user, assistant) exchange is used; later turns are
discarded. Instances where the first assistant turn contains no parseable
call are skipped entirely (not included as pseudo-abstention examples --
T3 is not BFCL's abstention tier, so we don't want to invent one from
parse failures).

Per project license policy, this tier is not redistributed -- it is loaded
directly from the upstream HF dataset at eval time, never checked into this
repo or re-uploaded to an HF dataset QuantCall controls.
"""

from __future__ import annotations

import random
import re
from typing import Any

from quantcall.datasets.base import NormalizedInstance, ToolCall, ToolSpec
from quantcall.datasets.bfcl import _extract_json_schema

_CALL_RE = re.compile(r"([A-Za-z][A-Za-z0-9_ .]*?)\(([^()]*)\)")
_KWARG_KEY_RE = re.compile(r"^([a-zA-Z_]\w*)\s*=\s*(.*)$", re.DOTALL)


def _extract_tool_list(system: str) -> list[dict[str, Any]] | None:
    """Pull the embedded JSON array of function defs out of the system prompt."""
    import json

    start = system.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(system)):
        ch = system[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = system[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, list) else None
    return None


def _tools_from_raw(raw_tools: list[dict[str, Any]]) -> list[ToolSpec]:
    tools: list[ToolSpec] = []
    for t in raw_tools:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        tools.append(
            ToolSpec(
                name=t["name"],
                description=t.get("description", ""),
                json_schema=_extract_json_schema(t.get("parameters") or {}),
            )
        )
    return tools


def _split_top_level_args(args_str: str) -> list[str]:
    """Split a Python-call argument string on top-level commas only.

    Naive comma-splitting breaks as soon as an argument value is itself a
    list/dict literal (e.g. ``marketTrends=[{"a": 1}, {"a": 2}]``), because
    the commas *inside* that literal aren't argument separators. This walks
    the string tracking bracket depth ([]/{}/()) and quote state, and only
    splits where depth is 0 and we're not inside a quoted string.
    """
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    i = 0
    while i < len(args_str):
        ch = args_str[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(args_str[start:i])
            start = i + 1
        i += 1
    tail = args_str[start:]
    if tail.strip():
        parts.append(tail)
    return parts


def _parse_python_kwargs(args_str: str) -> dict[str, Any]:
    import json as _json

    result: dict[str, Any] = {}
    for part in _split_top_level_args(args_str):
        m = _KWARG_KEY_RE.match(part.strip())
        if not m:
            continue
        key = m.group(1)
        raw_val = m.group(2).strip()
        if raw_val[:1] in "\"'" and raw_val[-1:] == raw_val[:1] and len(raw_val) >= 2:
            result[key] = raw_val[1:-1]
            continue
        try:
            result[key] = _json.loads(raw_val)
        except _json.JSONDecodeError:
            result[key] = raw_val
    return result


def _parse_call_list(text: str) -> list[ToolCall] | None:
    """Parse ToolACE's Python-call-list ground-truth format.

    Returns None (not an empty list) when nothing call-shaped is found, so
    callers can distinguish "no call in this text" from "parsed, zero calls".
    """
    text = text.strip()
    if not text.startswith("["):
        return None
    matches = list(_CALL_RE.finditer(text))
    if not matches:
        return None
    calls: list[ToolCall] = []
    for m in matches:
        name = m.group(1).strip()
        args = _parse_python_kwargs(m.group(2))
        calls.append(ToolCall(name=name, arguments=args))
    return calls


def _first_exchange(conversations: list[dict[str, Any]]) -> tuple[str, str] | None:
    query: str | None = None
    for turn in conversations:
        role = turn.get("from")
        if role == "user" and query is None:
            query = turn.get("value", "")
            continue
        if role == "assistant" and query is not None:
            return query, turn.get("value", "")
    return None


def normalize_toolace_row(raw: dict[str, Any], row_idx: int) -> NormalizedInstance | None:
    system = raw.get("system", "")
    raw_tools = _extract_tool_list(system)
    if not raw_tools:
        return None
    tools = _tools_from_raw(raw_tools)
    if not tools:
        return None

    exchange = _first_exchange(raw.get("conversations", []))
    if exchange is None:
        return None
    query, assistant_reply = exchange

    gt_calls = _parse_call_list(assistant_reply)
    if not gt_calls:
        return None

    return NormalizedInstance(
        id=f"T3-toolace-{row_idx}",
        tier="T3",
        category="toolace",
        query=query,
        tools=tools,
        ground_truth_calls=gt_calls,
        expects_call=True,
    )


def load_toolace(sample_size: int | None = 200, seed: int = 42) -> list[NormalizedInstance]:
    """Load ToolACE instances directly from the HF hub (not redistributed).

    Deterministically shuffles row order by `seed`, then keeps the first
    `sample_size` rows that parse cleanly (some rows are skipped -- see
    module docstring -- so more than `sample_size` raw rows may be scanned).
    """
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("Team-ACE/ToolACE", split="train")
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)

    instances: list[NormalizedInstance] = []
    for idx in order:
        inst = normalize_toolace_row(ds[idx], idx)
        if inst is not None:
            instances.append(inst)
        if sample_size is not None and len(instances) >= sample_size:
            break
    return instances

from __future__ import annotations

import json
from pathlib import Path

from quantcall.datasets.base import NormalizedInstance, ToolCall, ToolSpec

_SMOKE_PATH = Path(__file__).parent.parent.parent.parent / "data" / "smoke" / "t0_smoke.jsonl"


def load_smoke(path: str | Path | None = None) -> list[NormalizedInstance]:
    """Load the in-repo T0 smoke dataset (10 hand-crafted instances)."""
    p = Path(path) if path is not None else _SMOKE_PATH
    instances: list[NormalizedInstance] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            tools = [
                ToolSpec(
                    name=t["name"],
                    description=t.get("description", ""),
                    json_schema=t.get("json_schema", {}),
                )
                for t in raw.get("tools", [])
            ]
            gt_calls = [
                ToolCall(name=c["name"], arguments=c.get("arguments", {}))
                for c in raw.get("ground_truth_calls", [])
            ]
            instances.append(
                NormalizedInstance(
                    id=raw["id"],
                    tier=raw.get("tier", "T0"),
                    category=raw.get("category", "simple"),
                    query=raw["query"],
                    tools=tools,
                    ground_truth_calls=gt_calls,
                    expects_call=raw.get("expects_call", True),
                )
            )
    return instances

from __future__ import annotations

import hashlib
import json
from typing import Any

from quantcall.datasets.base import NormalizedInstance

NORMALIZATION_VERSION = "1.0"


def _instance_to_dict(inst: NormalizedInstance) -> dict[str, Any]:
    return {
        "id": inst.id,
        "tier": inst.tier,
        "category": inst.category,
        "query": inst.query,
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "json_schema": t.json_schema,
            }
            for t in inst.tools
        ],
        "ground_truth_calls": [
            {"name": c.name, "arguments": c.arguments} for c in inst.ground_truth_calls
        ],
        "expects_call": inst.expects_call,
    }


def content_sha256(inst: NormalizedInstance) -> str:
    """Deterministic SHA-256 of a normalized instance's content."""
    blob = json.dumps(_instance_to_dict(inst), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    json_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def __hash__(self) -> int:
        import json

        return hash((self.name, json.dumps(self.arguments, sort_keys=True)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return NotImplemented
        return self.name == other.name and self.arguments == other.arguments


@dataclass
class NormalizedInstance:
    id: str
    tier: str
    category: str
    query: str
    tools: list[ToolSpec] = field(default_factory=list)
    ground_truth_calls: list[ToolCall] = field(default_factory=list)
    expects_call: bool = True

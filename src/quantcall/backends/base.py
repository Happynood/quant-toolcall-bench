from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCallResult:
    raw_output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: float | None = None
    peak_vram_mb: float | None = None
    tokens_per_second: float | None = None


class Backend(ABC):
    @abstractmethod
    def generate_toolcall(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


def tools_to_openai_spec(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert ToolSpec list to OpenAI-compatible tool spec format."""
    from quantcall.datasets.base import ToolSpec

    result: list[dict[str, Any]] = []
    for t in tools:
        if isinstance(t, ToolSpec):
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.json_schema,
                    },
                }
            )
        else:
            result.append(t)
    return result

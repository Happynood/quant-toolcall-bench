from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantcall.datasets.base import NormalizedInstance, ToolSpec
from quantcall.datasets.bfcl import load_bfcl, normalize_bfcl_instance

SAMPLE_BFCL_SIMPLE = {
    "question_id": "simple_1",
    "question": "What is the weather in Paris?",
    "function": [
        {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        }
    ],
    "ground_truth": [{"get_weather": {"city": "Paris", "unit": "celsius"}}],
}

SAMPLE_BFCL_IRRELEVANCE = {
    "question_id": "irrel_1",
    "question": "Tell me a joke",
    "function": [
        {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ],
    "ground_truth": [],
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_normalize_bfcl_simple():
    inst = normalize_bfcl_instance(SAMPLE_BFCL_SIMPLE, category="simple")
    assert isinstance(inst, NormalizedInstance)
    assert inst.tier == "T1"
    assert inst.category == "simple"
    assert inst.query == "What is the weather in Paris?"
    assert len(inst.tools) == 1
    assert inst.tools[0].name == "get_weather"
    assert inst.expects_call is True
    assert len(inst.ground_truth_calls) == 1
    assert inst.ground_truth_calls[0].name == "get_weather"
    assert inst.ground_truth_calls[0].arguments["city"] == "Paris"


def test_normalize_bfcl_irrelevance():
    inst = normalize_bfcl_instance(SAMPLE_BFCL_IRRELEVANCE, category="irrelevance")
    assert inst.expects_call is False
    assert inst.ground_truth_calls == []
    assert inst.tier == "T6"


def test_tool_spec_has_json_schema():
    inst = normalize_bfcl_instance(SAMPLE_BFCL_SIMPLE, category="simple")
    tool = inst.tools[0]
    assert isinstance(tool, ToolSpec)
    assert "properties" in tool.json_schema
    assert "city" in tool.json_schema["properties"]


def test_load_bfcl_from_jsonl(tmp_path):
    jsonl = tmp_path / "simple.jsonl"
    _write_jsonl(jsonl, [SAMPLE_BFCL_SIMPLE, SAMPLE_BFCL_SIMPLE])
    instances = load_bfcl(
        categories=["simple"],
        data_dir=tmp_path,
        category_files={"simple": str(jsonl)},
    )
    assert len(instances) == 2
    assert all(isinstance(i, NormalizedInstance) for i in instances)


def test_load_bfcl_irrelevance_sets_tier_t6(tmp_path):
    jsonl = tmp_path / "irrelevance.jsonl"
    _write_jsonl(jsonl, [SAMPLE_BFCL_IRRELEVANCE])
    instances = load_bfcl(
        categories=["irrelevance"],
        data_dir=tmp_path,
        category_files={"irrelevance": str(jsonl)},
    )
    assert instances[0].tier == "T6"
    assert instances[0].expects_call is False


def test_load_bfcl_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bfcl(
            categories=["simple"],
            data_dir=tmp_path,
            category_files={"simple": str(tmp_path / "nonexistent.jsonl")},
        )

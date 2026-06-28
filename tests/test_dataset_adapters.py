from __future__ import annotations

from quantcall.datasets.base import NormalizedInstance
from quantcall.datasets.toolace import normalize_toolace_instance
from quantcall.datasets.xlam import normalize_xlam_instance

TOOLACE_SAMPLE = {
    "id": "ta_001",
    "query": "Search for flights from NYC to London",
    "answers": [
        {
            "name": "search_flights",
            "arguments": {"origin": "NYC", "destination": "LDN"},
        }
    ],
    "tools": [
        {
            "name": "search_flights",
            "description": "Search available flights",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                },
            },
        }
    ],
}

XLAM_SAMPLE = {
    "id": "xl_001",
    "query": "Get the stock price of AAPL",
    "answers": '[{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}]',
    "tools": (
        '[{"name": "get_stock_price", "description": "Get stock price",'
        ' "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}}}]'
    ),
}


def test_normalize_toolace_basic():
    inst = normalize_toolace_instance(TOOLACE_SAMPLE)
    assert isinstance(inst, NormalizedInstance)
    assert inst.tier == "T3"
    assert inst.category == "toolace"
    assert inst.expects_call is True
    assert len(inst.ground_truth_calls) == 1
    assert inst.ground_truth_calls[0].name == "search_flights"
    assert inst.ground_truth_calls[0].arguments["origin"] == "NYC"


def test_normalize_toolace_tools():
    inst = normalize_toolace_instance(TOOLACE_SAMPLE)
    assert len(inst.tools) == 1
    assert inst.tools[0].name == "search_flights"
    assert "properties" in inst.tools[0].json_schema


def test_normalize_xlam_basic():
    inst = normalize_xlam_instance(XLAM_SAMPLE)
    assert isinstance(inst, NormalizedInstance)
    assert inst.tier == "T4"
    assert inst.category == "xlam"
    assert inst.expects_call is True
    assert len(inst.ground_truth_calls) == 1
    assert inst.ground_truth_calls[0].name == "get_stock_price"


def test_normalize_xlam_json_string_tools():
    """xLAM tools field can be a JSON string."""
    inst = normalize_xlam_instance(XLAM_SAMPLE)
    assert len(inst.tools) == 1


def test_normalize_xlam_no_call():
    sample = dict(XLAM_SAMPLE)
    sample["answers"] = "[]"
    inst = normalize_xlam_instance(sample)
    assert inst.expects_call is False
    assert inst.ground_truth_calls == []


def test_normalize_toolace_no_answers():
    sample = dict(TOOLACE_SAMPLE)
    sample["answers"] = []
    inst = normalize_toolace_instance(sample)
    assert inst.expects_call is False

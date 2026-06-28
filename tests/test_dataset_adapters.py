from __future__ import annotations

from quantcall.config import QuantCallConfig
from quantcall.datasets.base import NormalizedInstance
from quantcall.datasets.toolace import normalize_toolace_instance
from quantcall.datasets.xlam import (
    XLAM_GATED_REPO,
    XLAM_UNGATED_REPO,
    normalize_xlam_instance,
    normalize_xlam_parsed_instance,
)

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

# Salesforce/gated format (flat fields)
XLAM_SAMPLE = {
    "id": "xl_001",
    "query": "Get the stock price of AAPL",
    "answers": '[{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}]',
    "tools": (
        '[{"name": "get_stock_price", "description": "Get stock price",'
        ' "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}}}]'
    ),
}

# minpeter/ungated format: {messages, tools (OpenAI-wrapped), extra}
XLAM_PARSED_SAMPLE = {
    "id": "xl_parsed_001",
    "messages": [{"role": "user", "content": "Get the stock price of AAPL"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get stock price",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
        }
    ],
    "extra": {"answers": '[{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}]'},
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
    inst = normalize_xlam_instance(XLAM_SAMPLE)
    assert len(inst.tools) == 1


def test_normalize_xlam_no_call():
    sample = dict(XLAM_SAMPLE)
    sample["answers"] = "[]"
    inst = normalize_xlam_instance(sample)
    assert inst.expects_call is False
    assert inst.ground_truth_calls == []


def test_normalize_xlam_parsed_basic():
    """Ungated minpeter format is parsed correctly."""
    inst = normalize_xlam_parsed_instance(XLAM_PARSED_SAMPLE)
    assert isinstance(inst, NormalizedInstance)
    assert inst.tier == "T4"
    assert inst.category == "xlam"
    assert inst.expects_call is True
    assert len(inst.ground_truth_calls) == 1
    assert inst.ground_truth_calls[0].name == "get_stock_price"
    assert inst.ground_truth_calls[0].arguments["ticker"] == "AAPL"


def test_normalize_xlam_parsed_openai_tool_format():
    """OpenAI-wrapped tool definitions are correctly unwrapped."""
    inst = normalize_xlam_parsed_instance(XLAM_PARSED_SAMPLE)
    assert len(inst.tools) == 1
    assert inst.tools[0].name == "get_stock_price"
    assert "properties" in inst.tools[0].json_schema
    assert inst.tools[0].json_schema.get("required") == ["ticker"]


def test_normalize_xlam_parsed_query_from_messages():
    """Query is extracted from the last user message."""
    sample = {
        "id": "q_test",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "tools": [],
        "extra": {"answers": "[]"},
    }
    inst = normalize_xlam_parsed_instance(sample)
    assert inst.query == "What is 2+2?"
    assert inst.expects_call is False


def test_normalize_xlam_parsed_no_call():
    sample = dict(XLAM_PARSED_SAMPLE)
    sample["extra"] = {"answers": "[]"}
    inst = normalize_xlam_parsed_instance(sample)
    assert inst.expects_call is False
    assert inst.ground_truth_calls == []


def test_normalize_toolace_no_answers():
    sample = dict(TOOLACE_SAMPLE)
    sample["answers"] = []
    inst = normalize_toolace_instance(sample)
    assert inst.expects_call is False


# --- Ungated-by-default policy ---


def test_default_config_uses_ungated_xlam():
    """Default config must NOT require a gated HF token for xLAM."""
    cfg = QuantCallConfig()
    assert cfg.use_gated_xlam is False


def test_xlam_ungated_repo_constant():
    """The ungated mirror repo ID is set correctly."""
    assert XLAM_UNGATED_REPO == "minpeter/xlam-function-calling-60k-parsed"
    assert XLAM_GATED_REPO == "Salesforce/xlam-function-calling-60k"


def test_default_tiers_ungated():
    """Default tier T0 requires no HF token. T4 is ungated when use_gated_xlam=False."""
    cfg = QuantCallConfig()
    # Default tier is T0 — always available, in-repo
    assert cfg.tiers == ["T0"]
    # Opt-in to T4 uses ungated mirror
    assert cfg.use_gated_xlam is False

from __future__ import annotations

from quantcall.datasets.base import NormalizedInstance
from quantcall.parsing.hermes_xml import HermesXmlParser
from quantcall.parsing.raw_json import RawJsonParser
from quantcall.runner import _build_messages, _get_parser


def _instance(query: str) -> NormalizedInstance:
    return NormalizedInstance(id="i0", tier="T1", category="simple", query=query)


def test_get_parser_default_is_raw_json():
    assert isinstance(_get_parser("default"), RawJsonParser)


def test_get_parser_qwen3_nothink_is_hermes_xml():
    assert isinstance(_get_parser("qwen3_nothink"), HermesXmlParser)


def test_build_messages_appends_nothink_for_qwen3_variant():
    instance = _instance("What's the weather in Paris?")
    messages = _build_messages(instance, "qwen3_nothink")
    assert messages[0]["content"] == "What's the weather in Paris? /no_think"


def test_build_messages_default_leaves_query_untouched():
    instance = _instance("What's the weather in Paris?")
    messages = _build_messages(instance, "default")
    assert messages[0]["content"] == "What's the weather in Paris?"


def test_hermes_parser_ignores_braces_inside_think_block():
    """A real Qwen3 raw output: <think> block full of JSON-looking noise, then a
    real tool_call. The Hermes parser must extract only the tool_call, not get
    confused by braces inside <think>...</think>."""
    raw_output = (
        "<think>\n"
        'The user wants {"weather": true} for Paris. Let me consider '
        'options like {"city": "Paris", "unit": "celsius"} before deciding.\n'
        "</think>\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", '
        '"unit": "celsius"}}\n</tool_call>'
    )
    parser = HermesXmlParser()
    calls = parser.parse(raw_output)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Paris", "unit": "celsius"}


def test_hermes_parser_returns_empty_for_pure_think_no_tool_call():
    raw_output = "<think>\nI don't think any tool is needed here.\n</think>\nNo tool needed."
    parser = HermesXmlParser()
    assert parser.parse(raw_output) == []

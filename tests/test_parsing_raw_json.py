from __future__ import annotations

from quantcall.parsing.base import ParsedCall
from quantcall.parsing.raw_json import RawJsonParser


def _parser() -> RawJsonParser:
    return RawJsonParser()


def test_parse_tool_call_wrapper():
    p = _parser()
    raw = '{"tool_call": {"name": "get_weather", "arguments": {"city": "Paris"}}}'
    calls = p.parse(raw)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Paris"}


def test_parse_flat_name_arguments():
    p = _parser()
    raw = '{"name": "web_search", "arguments": {"query": "python tutorials"}}'
    calls = p.parse(raw)
    assert len(calls) == 1
    assert calls[0].name == "web_search"


def test_parse_no_tool_call_in_text():
    p = _parser()
    calls = p.parse("I don't need to call any function here.")
    assert calls == []


def test_parse_embedded_in_text():
    p = _parser()
    raw = 'I will call: {"tool_call": {"name": "calculate", "arguments": {"expr": "2+2"}}}'
    calls = p.parse(raw)
    assert len(calls) == 1
    assert calls[0].name == "calculate"


def test_parse_json_code_block():
    p = _parser()
    raw = '```json\n{"tool_call": {"name": "translate_text", "arguments": {}}}\n```'
    calls = p.parse(raw)
    assert len(calls) == 1
    assert calls[0].name == "translate_text"


def test_parse_empty_string():
    assert _parser().parse("") == []


def test_parser_name():
    assert _parser().name == "raw_json"


def test_parsed_call_equality():
    a = ParsedCall(name="foo", arguments={"x": 1})
    b = ParsedCall(name="foo", arguments={"x": 1})
    c = ParsedCall(name="foo", arguments={"x": 2})
    assert a == b
    assert a != c

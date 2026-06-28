from __future__ import annotations

from quantcall.parsing.hermes_xml import HermesXmlParser
from quantcall.parsing.xlam_parser import XlamParser


def test_hermes_xml_basic():
    p = HermesXmlParser()
    raw = '<tool_call>\n{"name": "search", "arguments": {"q": "hello"}}\n</tool_call>'
    calls = p.parse(raw)
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"q": "hello"}


def test_hermes_xml_multiple_calls():
    p = HermesXmlParser()
    raw = (
        '<tool_call>\n{"name": "fn_a", "arguments": {"x": 1}}\n</tool_call>\n'
        '<tool_call>\n{"name": "fn_b", "arguments": {"y": 2}}\n</tool_call>'
    )
    calls = p.parse(raw)
    assert len(calls) == 2
    assert {c.name for c in calls} == {"fn_a", "fn_b"}


def test_hermes_xml_empty():
    p = HermesXmlParser()
    assert p.parse("I cannot help with that.") == []


def test_hermes_xml_name():
    assert HermesXmlParser().name == "hermes_xml"


def test_xlam_parser_json_list():
    p = XlamParser()
    raw = '[{"name": "get_price", "arguments": {"ticker": "AAPL"}}]'
    calls = p.parse(raw)
    assert len(calls) == 1
    assert calls[0].name == "get_price"
    assert calls[0].arguments == {"ticker": "AAPL"}


def test_xlam_parser_wrapped():
    p = XlamParser()
    raw = '{"tool_calls": [{"name": "fn", "arguments": {"k": "v"}}]}'
    calls = p.parse(raw)
    assert len(calls) == 1
    assert calls[0].name == "fn"


def test_xlam_parser_empty():
    p = XlamParser()
    assert p.parse("[]") == []
    assert p.parse("") == []


def test_xlam_parser_name():
    assert XlamParser().name == "xlam"

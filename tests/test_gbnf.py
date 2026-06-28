from __future__ import annotations

from quantcall.decoding.gbnf import gbnf_from_schema


def test_empty_object_schema():
    grammar = gbnf_from_schema({"type": "object", "properties": {}})
    assert "root" in grammar
    assert "{" in grammar


def test_string_property():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    grammar = gbnf_from_schema(schema)
    assert "string" in grammar
    assert "city" in grammar


def test_integer_property():
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
    }
    grammar = gbnf_from_schema(schema)
    assert "integer" in grammar or "number" in grammar


def test_boolean_property():
    schema = {"type": "object", "properties": {"active": {"type": "boolean"}}}
    grammar = gbnf_from_schema(schema)
    assert "true" in grammar or "boolean" in grammar


def test_enum_property():
    schema = {
        "type": "object",
        "properties": {"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
    }
    grammar = gbnf_from_schema(schema)
    assert "celsius" in grammar
    assert "fahrenheit" in grammar


def test_nested_object():
    schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }
        },
    }
    grammar = gbnf_from_schema(schema)
    assert "location" in grammar
    assert "string" in grammar


def test_array_property():
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    }
    grammar = gbnf_from_schema(schema)
    assert "array" in grammar or "[" in grammar


def test_grammar_is_string():
    grammar = gbnf_from_schema({"type": "object", "properties": {}})
    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_multi_property_schema():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
    }
    grammar = gbnf_from_schema(schema)
    assert "name" in grammar
    assert "age" in grammar

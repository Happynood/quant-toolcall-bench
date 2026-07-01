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


def test_build_tool_call_grammar_single_tool():
    from quantcall.decoding.gbnf import build_tool_call_grammar

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    grammar = build_tool_call_grammar(tools)
    assert "get_weather" in grammar
    assert "<tool_call>" in grammar
    assert "</tool_call>" in grammar
    assert "city" in grammar


def test_build_tool_call_grammar_multiple_tools_alternation():
    from quantcall.decoding.gbnf import build_tool_call_grammar

    tools = [
        {"type": "function", "function": {"name": "tool_a", "parameters": {}}},
        {"type": "function", "function": {"name": "tool_b", "parameters": {}}},
    ]
    grammar = build_tool_call_grammar(tools)
    assert "tool_a" in grammar
    assert "tool_b" in grammar
    assert "|" in grammar  # alternation between the two tools


def test_build_tool_call_grammar_empty_tools_falls_back_to_generic_value():
    from quantcall.decoding.gbnf import build_tool_call_grammar

    grammar = build_tool_call_grammar([])
    assert "tool-call-body ::= value" in grammar


def test_build_tool_call_grammar_is_valid_gbnf_via_llama_cpp():
    """Real compile check: skip if the llama-cpp extra / CUDA libs aren't
    available in this environment (optional extra, not required for the
    rest of the test suite), but if it IS available, actually compile the
    grammar through LlamaGrammar to catch real GBNF syntax errors."""
    import pytest

    try:
        from quantcall.backends.llama_cpp import _preload_cuda_libs

        _preload_cuda_libs()
        from llama_cpp import LlamaGrammar  # type: ignore[import]
    except Exception:
        pytest.skip("llama-cpp extra / CUDA libs not available in this environment")

    from quantcall.decoding.gbnf import build_tool_call_grammar

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    grammar_text = build_tool_call_grammar(tools)
    compiled = LlamaGrammar.from_string(grammar_text, verbose=False)
    assert compiled is not None


def test_schema_rule_name_never_mixes_underscore_and_hyphen():
    """Regression test: llama.cpp's GBNF parser segfaults on rule names that
    mix "_" and "-" (e.g. "product_list-array" crashes the process outright;
    reproduced directly against llama-cpp-python with a real Qwen3 model).
    JSON Schema property names commonly contain underscores, and the array
    rule-naming path appended "-array", so every array property produced a
    mixed-separator name. All separators must normalize to "-"."""
    from quantcall.decoding.gbnf import _schema_rule_name

    name = _schema_rule_name("tool1.product_list")
    assert "_" not in name
    assert name == "tool1-product-list"


def test_build_tool_call_grammar_array_property_has_no_mixed_separators():
    from quantcall.decoding.gbnf import build_tool_call_grammar

    tools = [
        {
            "type": "function",
            "function": {
                "name": "walmart.purchase",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_list": {"type": "array"},
                        "pack_size": {"type": "array"},
                    },
                    "required": ["product_list"],
                },
            },
        }
    ]
    grammar = build_tool_call_grammar(tools)
    import re

    for line in grammar.splitlines():
        m = re.match(r"^([a-zA-Z0-9_-]+)\s*::=", line)
        if m:
            name = m.group(1)
            assert not ("_" in name and "-" in name), f"mixed-separator rule name: {name}"

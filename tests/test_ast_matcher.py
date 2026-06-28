from __future__ import annotations

from quantcall.validation.ast_matcher import ast_equal, normalize_value


def test_exact_match():
    assert ast_equal({"city": "Paris"}, {"city": "Paris"}) is True


def test_type_coercion_int_float():
    assert ast_equal({"n": 1}, {"n": 1.0}) is True


def test_type_coercion_string_int():
    assert ast_equal({"n": "42"}, {"n": 42}) is True


def test_nested_dict_match():
    pred = {"location": {"city": "Paris", "country": "FR"}}
    gt = {"location": {"city": "Paris", "country": "FR"}}
    assert ast_equal(pred, gt) is True


def test_nested_dict_mismatch():
    assert ast_equal({"location": {"city": "Paris"}}, {"location": {"city": "Berlin"}}) is False


def test_list_match():
    assert ast_equal({"items": [1, 2, 3]}, {"items": [1, 2, 3]}) is True


def test_list_order_matters():
    assert ast_equal({"items": [1, 2]}, {"items": [2, 1]}) is False


def test_boolean_coercion():
    assert ast_equal({"flag": "true"}, {"flag": True}) is True
    assert ast_equal({"flag": "false"}, {"flag": False}) is True


def test_none_vs_null():
    assert ast_equal({"x": None}, {"x": None}) is True


def test_extra_key_in_pred_not_in_gt():
    assert ast_equal({"a": 1, "b": 2}, {"a": 1}) is False


def test_missing_key_in_pred():
    assert ast_equal({"a": 1}, {"a": 1, "b": 2}) is False


def test_normalize_numeric_string():
    assert normalize_value("3.14") == 3.14
    assert normalize_value("42") == 42


def test_normalize_bool_string():
    assert normalize_value("true") is True
    assert normalize_value("false") is False
    assert normalize_value("True") is True

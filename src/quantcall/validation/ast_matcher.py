from __future__ import annotations

from typing import Any


def normalize_value(val: Any) -> Any:
    """Coerce strings to their canonical Python types where safe."""
    if isinstance(val, str):
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
        try:
            int_val = int(val)
            return int_val
        except ValueError:
            pass
        try:
            float_val = float(val)
            return float_val
        except ValueError:
            pass
    return val


def ast_equal(predicted: Any, ground_truth: Any) -> bool:
    """Recursively compare two values with type coercion."""
    if isinstance(ground_truth, dict) and isinstance(predicted, dict):
        if set(predicted.keys()) != set(ground_truth.keys()):
            return False
        return all(ast_equal(predicted[k], ground_truth[k]) for k in ground_truth)

    if isinstance(ground_truth, list) and isinstance(predicted, list):
        if len(predicted) != len(ground_truth):
            return False
        return all(ast_equal(p, g) for p, g in zip(predicted, ground_truth, strict=True))

    norm_pred = normalize_value(predicted)
    norm_gt = normalize_value(ground_truth)

    if isinstance(norm_gt, float) and isinstance(norm_pred, (int, float)):
        return abs(float(norm_pred) - norm_gt) < 1e-9
    if isinstance(norm_pred, float) and isinstance(norm_gt, (int, float)):
        return abs(norm_pred - float(norm_gt)) < 1e-9

    return norm_pred == norm_gt

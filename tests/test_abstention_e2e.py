from __future__ import annotations

from quantcall.datasets.base import NormalizedInstance, ToolSpec
from quantcall.metrics.core import compute_metrics, evaluate_instance
from quantcall.parsing.base import ParsedCall


def _make_no_call_instance(inst_id: str = "test-abs-1") -> NormalizedInstance:
    return NormalizedInstance(
        id=inst_id,
        tier="T6",
        category="irrelevance",
        query="Tell me a joke",
        tools=[
            ToolSpec(
                name="get_weather",
                description="Get weather",
                json_schema={"type": "object", "properties": {}},
            )
        ],
        ground_truth_calls=[],
        expects_call=False,
    )


def test_correct_abstention_scores_full():
    """Model correctly declines → abstention=1.0."""
    inst = _make_no_call_instance()
    result = evaluate_instance(inst, predicted_calls=[], parse_succeeded=True)
    metrics = compute_metrics([result])
    assert metrics.abstention == 1.0


def test_over_call_scores_zero():
    """Model emits a call when none expected → abstention=0.0, over_call=1.0."""
    inst = _make_no_call_instance()
    predicted = [ParsedCall(name="get_weather", arguments={"city": "Paris"})]
    result = evaluate_instance(inst, predicted_calls=predicted, parse_succeeded=True)
    metrics = compute_metrics([result])
    assert metrics.abstention == 0.0
    assert metrics.over_call == 1.0


def test_mixed_abstention():
    """2 instances: one correct abstention, one over-call → abstention=0.5."""
    inst1 = _make_no_call_instance("abs-1")
    inst2 = _make_no_call_instance("abs-2")
    r1 = evaluate_instance(inst1, predicted_calls=[], parse_succeeded=True)
    r2 = evaluate_instance(
        inst2,
        predicted_calls=[ParsedCall(name="get_weather", arguments={})],
        parse_succeeded=True,
    )
    metrics = compute_metrics([r1, r2])
    assert abs(metrics.abstention - 0.5) < 1e-9

from __future__ import annotations

from quantcall.datasets.base import NormalizedInstance, ToolCall, ToolSpec
from quantcall.metrics.core import (
    FcrWeights,
    compute_metrics,
    evaluate_instance,
)
from quantcall.parsing.base import ParsedCall


def _make_instance(
    expects_call: bool = True,
    gt_name: str = "get_weather",
    gt_args: dict | None = None,
) -> NormalizedInstance:
    return NormalizedInstance(
        id="T0-test",
        tier="T0",
        category="simple",
        query="What's the weather?",
        tools=[
            ToolSpec(
                name="get_weather",
                description="Get weather",
                json_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            )
        ],
        ground_truth_calls=(
            [ToolCall(name=gt_name, arguments=gt_args or {"city": "Paris"})] if expects_call else []
        ),
        expects_call=expects_call,
    )


def test_evaluate_instance_schema_valid():
    inst = _make_instance()
    pred = [ParsedCall(name="get_weather", arguments={"city": "Paris"})]
    result = evaluate_instance(inst, pred, parse_succeeded=True)
    assert result.schema_valid is True
    assert result.names_exact_match is True
    assert result.args_correct is True


def test_evaluate_instance_schema_invalid_wrong_type():
    inst = _make_instance()
    pred = [ParsedCall(name="get_weather", arguments={"city": 42})]
    result = evaluate_instance(inst, pred, parse_succeeded=True)
    assert result.schema_valid is False


def test_evaluate_instance_parse_failed():
    inst = _make_instance()
    result = evaluate_instance(inst, [], parse_succeeded=False)
    assert result.parse_succeeded is False
    assert result.schema_valid is False


def test_evaluate_instance_no_call_when_expected():
    inst = _make_instance(expects_call=True)
    result = evaluate_instance(inst, [], parse_succeeded=True)
    assert result.names_exact_match is False
    assert result.schema_valid is True


def test_evaluate_instance_correct_abstention():
    inst = _make_instance(expects_call=False)
    result = evaluate_instance(inst, [], parse_succeeded=True)
    assert result.names_exact_match is True
    assert result.schema_valid is True


def test_compute_metrics_all_correct():
    inst = _make_instance(expects_call=True, gt_args={"city": "Paris"})
    pred = [ParsedCall(name="get_weather", arguments={"city": "Paris"})]
    ir = evaluate_instance(inst, pred, parse_succeeded=True)
    metrics = compute_metrics([ir])
    assert metrics.svr == 1.0
    assert metrics.tsa == 1.0
    assert metrics.ac == 1.0
    assert metrics.n == 1


def test_compute_metrics_all_wrong():
    inst = _make_instance()
    ir = evaluate_instance(inst, [], parse_succeeded=False)
    metrics = compute_metrics([ir])
    assert metrics.svr == 0.0
    assert metrics.n == 1


def test_compute_metrics_abstention():
    inst_no_call = _make_instance(expects_call=False)
    ir = evaluate_instance(inst_no_call, [], parse_succeeded=True)
    metrics = compute_metrics([ir])
    assert metrics.abstention == 1.0
    assert metrics.over_call == 0.0


def test_compute_metrics_over_call():
    inst_no_call = _make_instance(expects_call=False)
    ir = evaluate_instance(
        inst_no_call,
        [ParsedCall(name="get_weather", arguments={})],
        parse_succeeded=True,
    )
    metrics = compute_metrics([ir])
    assert metrics.over_call == 1.0
    assert metrics.abstention == 0.0


def test_compute_metrics_fcr_weighted():
    inst = _make_instance(expects_call=True, gt_args={})
    pred = [ParsedCall(name="get_weather", arguments={})]
    ir = evaluate_instance(inst, pred, parse_succeeded=True)
    weights = FcrWeights(svr=0.5, tsa=0.5, ac=0.0, abst=0.0)
    metrics = compute_metrics([ir], weights)
    assert 0.0 <= metrics.fcr <= 1.0


def test_compute_metrics_empty():
    metrics = compute_metrics([])
    assert metrics.n == 0
    assert metrics.svr == 0.0
    assert metrics.fcr == 0.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantcall.datasets.base import NormalizedInstance, ToolCall
from quantcall.parsing.base import ParsedCall
from quantcall.validation.schema_validator import validate_call


@dataclass
class InstanceResult:
    instance_id: str
    tier: str
    category: str
    expects_call: bool
    predicted_calls: list[ParsedCall]
    ground_truth_names: frozenset[str]
    parse_succeeded: bool
    schema_valid: bool
    names_exact_match: bool
    args_correct: bool


@dataclass
class FcrWeights:
    svr: float = 0.25
    tsa: float = 0.25
    ac: float = 0.25
    abst: float = 0.25


@dataclass
class MetricsResult:
    n: int
    svr: float
    tsa: float
    tsa_precision: float
    tsa_recall: float
    ac: float
    abstention: float
    over_call: float
    fcr: float
    instance_results: list[InstanceResult] = field(default_factory=list)


def _names_exact_match(predicted: list[ParsedCall], ground_truth: list[ToolCall]) -> bool:
    pred_names = {c.name for c in predicted}
    gt_names = {c.name for c in ground_truth}
    return pred_names == gt_names


def _names_precision(predicted: list[ParsedCall], ground_truth: list[ToolCall]) -> float:
    if not predicted:
        return 1.0 if not ground_truth else 0.0
    pred_names = {c.name for c in predicted}
    gt_names = {c.name for c in ground_truth}
    return len(pred_names & gt_names) / len(pred_names)


def _names_recall(predicted: list[ParsedCall], ground_truth: list[ToolCall]) -> float:
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    pred_names = {c.name for c in predicted}
    gt_names = {c.name for c in ground_truth}
    return len(pred_names & gt_names) / len(gt_names)


def _args_correct(predicted: list[ParsedCall], ground_truth: list[ToolCall]) -> bool:
    """Simple argument equality check (full AST matching added in Phase 1)."""
    if len(predicted) != len(ground_truth):
        return False
    for pc in predicted:
        matched_gt = [g for g in ground_truth if g.name == pc.name]
        if not matched_gt:
            return False
        if not any(pc.arguments == g.arguments for g in matched_gt):
            return False
    return True


def evaluate_instance(
    instance: NormalizedInstance,
    predicted_calls: list[ParsedCall],
    parse_succeeded: bool,
) -> InstanceResult:
    tool_schemas: dict[str, dict[str, Any]] = {t.name: t.json_schema for t in instance.tools}

    if not parse_succeeded:
        schema_valid = False
    elif not predicted_calls:
        schema_valid = True
    else:
        schema_valid = all(
            call.name in tool_schemas and validate_call(call, tool_schemas[call.name])
            for call in predicted_calls
        )

    names_match = _names_exact_match(predicted_calls, instance.ground_truth_calls)
    args_ok = _args_correct(predicted_calls, instance.ground_truth_calls)

    return InstanceResult(
        instance_id=instance.id,
        tier=instance.tier,
        category=instance.category,
        expects_call=instance.expects_call,
        predicted_calls=predicted_calls,
        ground_truth_names=frozenset(c.name for c in instance.ground_truth_calls),
        parse_succeeded=parse_succeeded,
        schema_valid=schema_valid,
        names_exact_match=names_match,
        args_correct=args_ok,
    )


def compute_metrics(
    instance_results: list[InstanceResult],
    weights: FcrWeights | None = None,
) -> MetricsResult:
    if weights is None:
        weights = FcrWeights()

    n = len(instance_results)
    if n == 0:
        return MetricsResult(
            n=0,
            svr=0.0,
            tsa=0.0,
            tsa_precision=0.0,
            tsa_recall=0.0,
            ac=0.0,
            abstention=0.0,
            over_call=0.0,
            fcr=0.0,
        )

    svr = sum(1 for r in instance_results if r.schema_valid) / n
    tsa = sum(1 for r in instance_results if r.names_exact_match) / n

    precisions = []
    recalls = []
    for r in instance_results:
        pred_names = {c.name for c in r.predicted_calls}
        gt_names: set[str] = set(r.ground_truth_names)

        if pred_names:
            p = len(pred_names & gt_names) / len(pred_names)
        else:
            p = 1.0 if not gt_names else 0.0
        if gt_names:
            rec = len(pred_names & gt_names) / len(gt_names)
        else:
            rec = 1.0 if not pred_names else 0.0
        precisions.append(p)
        recalls.append(rec)

    tsa_precision = sum(precisions) / n
    tsa_recall = sum(recalls) / n

    matched = [r for r in instance_results if r.names_exact_match]
    ac = sum(1 for r in matched if r.args_correct) / len(matched) if matched else 0.0

    abstention_instances = [r for r in instance_results if not r.expects_call]
    if abstention_instances:
        abstention = sum(1 for r in abstention_instances if not r.predicted_calls) / len(
            abstention_instances
        )
    else:
        abstention = 1.0

    over_call = 1.0 - abstention

    fcr = weights.svr * svr + weights.tsa * tsa + weights.ac * ac + weights.abst * abstention

    return MetricsResult(
        n=n,
        svr=svr,
        tsa=tsa,
        tsa_precision=tsa_precision,
        tsa_recall=tsa_recall,
        ac=ac,
        abstention=abstention,
        over_call=over_call,
        fcr=fcr,
        instance_results=instance_results,
    )

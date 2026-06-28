from __future__ import annotations

from quantcall.datasets.base import NormalizedInstance, ToolSpec
from quantcall.datasets.smoke import load_smoke


def test_load_smoke_returns_10_instances():
    instances = load_smoke()
    assert len(instances) == 10


def test_smoke_instances_are_normalized():
    for inst in load_smoke():
        assert isinstance(inst, NormalizedInstance)
        assert inst.id.startswith("T0-")
        assert inst.tier == "T0"
        assert inst.query
        assert isinstance(inst.tools, list)
        assert len(inst.tools) >= 1


def test_smoke_tools_have_schema():
    for inst in load_smoke():
        for tool in inst.tools:
            assert isinstance(tool, ToolSpec)
            assert tool.name
            assert isinstance(tool.json_schema, dict)


def test_smoke_expects_call_distribution():
    instances = load_smoke()
    call_expected = [i for i in instances if i.expects_call]
    no_call = [i for i in instances if not i.expects_call]
    assert len(call_expected) >= 1
    assert len(no_call) >= 1


def test_smoke_ground_truth_calls_match_expects_call():
    for inst in load_smoke():
        if inst.expects_call:
            assert len(inst.ground_truth_calls) >= 1
        else:
            assert len(inst.ground_truth_calls) == 0


def test_smoke_from_custom_path(smoke_jsonl_path):
    instances = load_smoke(path=smoke_jsonl_path)
    assert len(instances) == 10

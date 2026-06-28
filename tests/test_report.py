from __future__ import annotations

import json

from quantcall.report.leaderboard import build_leaderboard
from quantcall.report.tables import render_delta_table, render_metrics_table

RESULT_A = {
    "svr": 0.9,
    "tsa": 0.85,
    "ac": 0.7,
    "abstention": 0.8,
    "fcr": 0.8125,
    "n": 100,
    "config": {"model": "qwen3", "quant": "fp16", "backend": "llama-cpp"},
}
RESULT_B = {
    "svr": 0.7,
    "tsa": 0.65,
    "ac": 0.5,
    "abstention": 0.6,
    "fcr": 0.6125,
    "n": 100,
    "config": {"model": "qwen3", "quant": "Q4_K_M", "backend": "llama-cpp"},
}


def test_render_metrics_table():
    table = render_metrics_table([RESULT_A, RESULT_B])
    assert "SVR" in table
    assert "FCR" in table
    assert "qwen3" in table
    assert "fp16" in table
    assert "Q4_K_M" in table


def test_render_delta_table():
    table = render_delta_table(baseline=RESULT_A, current=RESULT_B)
    assert "ΔSVR" in table or "Delta" in table or "delta" in table.lower()
    assert "Q4_K_M" in table


def test_build_leaderboard_from_dir(tmp_path):
    for i, r in enumerate([RESULT_A, RESULT_B]):
        p = tmp_path / f"result_{i}.json"
        p.write_text(json.dumps(r))

    lb = build_leaderboard(results_dir=tmp_path)
    assert isinstance(lb, dict)
    assert "rows" in lb
    assert len(lb["rows"]) == 2


def test_build_leaderboard_writes_files(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "r1.json").write_text(json.dumps(RESULT_A))

    out_dir = tmp_path / "leaderboard"
    out_dir.mkdir()
    build_leaderboard(results_dir=results_dir, output_dir=out_dir)

    assert (out_dir / "leaderboard.json").exists()
    assert (out_dir / "leaderboard.csv").exists()
    assert (out_dir / "leaderboard.md").exists()


def test_leaderboard_json_valid(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "r.json").write_text(json.dumps(RESULT_A))

    out_dir = tmp_path / "lb"
    out_dir.mkdir()
    build_leaderboard(results_dir=results_dir, output_dir=out_dir)

    data = json.loads((out_dir / "leaderboard.json").read_text())
    assert "rows" in data
    assert "generated_at" in data

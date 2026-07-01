from __future__ import annotations

import json
import re
from pathlib import Path

from quantcall.report.leaderboard import build_leaderboard
from quantcall.report.published import (
    LEADERBOARD_COLS,
    RUNS_COLS,
    aggregate_leaderboard,
    run_row,
)
from quantcall.report.tables import render_delta_table

SCHEMA_DOC = Path(__file__).parent.parent / "docs" / "RESULTS_SCHEMA.md"
CARD_DOC = Path(__file__).parent.parent / "docs" / "hf" / "results_dataset_card.md"

RESULT_A = {
    "svr": 0.9,
    "tsa": 0.85,
    "ac": 0.7,
    "abstention": 0.8,
    "fcr": 0.8125,
    "n": 100,
    "vram_gb": 6.0,
    "config": {
        "model": "qwen3",
        "quant": "fp16",
        "backend": "llama-cpp",
        "decoding": "free",
        "tiers": ["T1"],
        "seed": 42,
        "sample_size": 100,
    },
    "manifest": {
        "git_commit": "abc123",
        "config_sha256": "cfg1",
        "dataset_sha256": "ds1",
        "timestamp": "2026-07-01T00:00:00+00:00",
    },
}
RESULT_B = {
    "svr": 0.7,
    "tsa": 0.65,
    "ac": 0.5,
    "abstention": 0.6,
    "fcr": 0.6125,
    "n": 100,
    "vram_gb": 3.2,
    "config": {
        "model": "qwen3",
        "quant": "Q4_K_M",
        "backend": "llama-cpp",
        "decoding": "free",
        "tiers": ["T1"],
        "seed": 42,
        "sample_size": 100,
    },
    "manifest": {
        "git_commit": "abc123",
        "config_sha256": "cfg2",
        "dataset_sha256": "ds1",
        "timestamp": "2026-07-01T00:01:00+00:00",
    },
}


def test_run_row_flattens_config_and_manifest():
    row = run_row(RESULT_A)
    assert row["model"] == "qwen3"
    assert row["quant"] == "fp16"
    assert row["tier"] == "T1"
    assert row["seed"] == 42
    assert row["vram_gb"] == 6.0
    assert row["git_commit"] == "abc123"


def test_aggregate_leaderboard_picks_fp16_baseline():
    rows = [run_row(RESULT_A), run_row(RESULT_B)]
    agg = aggregate_leaderboard(rows)
    assert len(agg) == 2

    by_quant = {r["quant"]: r for r in agg}
    assert by_quant["fp16"]["baseline_quant"] == "fp16"
    assert by_quant["Q4_K_M"]["baseline_quant"] == "fp16"
    assert by_quant["fp16"]["delta_fcr_rel"] is None
    assert by_quant["Q4_K_M"]["delta_fcr_rel"] is not None
    assert by_quant["Q4_K_M"]["delta_fcr_rel"] > 0  # degraded relative to fp16


def test_aggregate_leaderboard_labels_fallback_baseline_when_no_fp16():
    b_only = [run_row(RESULT_B)]
    agg = aggregate_leaderboard(b_only)
    assert len(agg) == 1
    assert agg[0]["baseline_quant"] == "Q4_K_M"
    assert agg[0]["delta_fcr_rel"] is None


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

    assert (out_dir / "runs.csv").exists()
    assert (out_dir / "leaderboard.json").exists()
    assert (out_dir / "leaderboard.csv").exists()
    assert (out_dir / "leaderboard.md").exists()


def test_runs_csv_has_seed_column(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "r1.json").write_text(json.dumps(RESULT_A))

    out_dir = tmp_path / "leaderboard"
    build_leaderboard(results_dir=results_dir, output_dir=out_dir)

    header = (out_dir / "runs.csv").read_text().splitlines()[0]
    assert "seed" in header.split(",")


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


def _extract_markdown_table_columns(markdown: str, heading: str) -> list[str]:
    """Pull the `Column` values out of the markdown table under `heading`."""
    section = markdown.split(heading, 1)[1]
    next_heading = re.search(r"\n## ", section)
    if next_heading:
        section = section[: next_heading.start()]
    cols = []
    for line in section.splitlines():
        m = re.match(r"\|\s*`(\w+)`\s*\|", line)
        if m:
            cols.append(m.group(1))
    return cols


def test_no_schema_drift():
    """docs/RESULTS_SCHEMA.md must document exactly the columns the code emits."""
    doc = SCHEMA_DOC.read_text()

    documented_runs = _extract_markdown_table_columns(doc, "## `runs.csv`")
    documented_leaderboard = _extract_markdown_table_columns(doc, "## `leaderboard.csv`")

    assert documented_runs == RUNS_COLS
    assert documented_leaderboard == LEADERBOARD_COLS


def test_hf_dataset_card_schema_matches_code():
    """The published HF dataset card must document exactly the columns the code emits."""
    card = CARD_DOC.read_text()

    assert "currently empty" not in card.lower()
    assert "happynood/quantcall-suite" in card
    assert "happynood/quantcall-results" in card

    documented_runs = _extract_markdown_table_columns(card, "## Schema: `data/runs.csv`")
    documented_leaderboard = _extract_markdown_table_columns(
        card, "## Schema: `data/leaderboard.csv`"
    )

    assert documented_runs == RUNS_COLS
    assert documented_leaderboard == LEADERBOARD_COLS

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quantcall.report.tables import _COLS, _row_from_result, render_metrics_table


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            if "svr" in data:
                results.append(data)
        except (json.JSONDecodeError, OSError):
            pass
    return results


def build_leaderboard(
    results_dir: Path | str,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    results = _load_results(results_dir)

    rows = [_row_from_result(r) for r in results]
    leaderboard: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "columns": _COLS,
        "rows": rows,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))

        with open(out / "leaderboard.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_COLS)
            writer.writerows(rows)

        md = render_metrics_table(results) if results else "No results yet."
        (out / "leaderboard.md").write_text(f"# Leaderboard\n\n{md}\n")

    return leaderboard

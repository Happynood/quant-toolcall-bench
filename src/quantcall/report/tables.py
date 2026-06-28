from __future__ import annotations

from typing import Any

from quantcall.metrics.deltas import compute_delta, compute_delta_rel

_COLS = ["Model", "Quant", "Backend", "SVR", "TSA", "AC", "Abst", "FCR"]


def _row_from_result(r: dict[str, Any]) -> list[str]:
    cfg = r.get("config", {})
    return [
        cfg.get("model", "—"),
        cfg.get("quant", "—"),
        cfg.get("backend", "—"),
        f"{r.get('svr', 0):.3f}",
        f"{r.get('tsa', 0):.3f}",
        f"{r.get('ac', 0):.3f}",
        f"{r.get('abstention', 0):.3f}",
        f"{r.get('fcr', 0):.3f}",
    ]


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = ["| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(row)) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def render_metrics_table(results: list[dict[str, Any]]) -> str:
    rows = [_row_from_result(r) for r in results]
    return _md_table(_COLS, rows)


def render_delta_table(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    cfg = current.get("config", {})
    rows: list[list[str]] = []
    for metric in ("svr", "tsa", "ac", "abstention", "fcr"):
        base_val = float(baseline.get(metric, 0))
        curr_val = float(current.get(metric, 0))
        delta = compute_delta(base_val, curr_val)
        delta_rel = compute_delta_rel(base_val, curr_val)
        rel_str = f"{delta_rel:.3f}" if delta_rel is not None else "—"
        rows.append(
            [
                f"Δ{metric.upper()}",
                cfg.get("model", "—"),
                cfg.get("quant", "—"),
                cfg.get("backend", "—"),
                f"{delta:+.3f}",
                rel_str,
            ]
        )
    headers = ["Metric", "Model", "Quant", "Backend", "ΔFCR", "Δrel"]
    return _md_table(headers, rows)

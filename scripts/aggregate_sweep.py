"""Aggregate Phase 4 sweep results across seeds and compute bootstrap CIs.

Reads results/sweep/{quant}_s{seed}.json for each quant level, computes the
mean and a bootstrap 95% CI (resampled across the 3 per-seed run-level means),
and reports the delta against the reference precision (Q8_0 — fp16 does not
fit the 4GB VRAM budget on this hardware, so Q8_0 is the least-lossy
precision we could run and is used as the Δ reference).

All numbers here come directly from results/sweep/*.json, which were written
by real `quantcall run` executions against the llama-cpp backend. Nothing in
this script fabricates or estimates a metric value.
"""

from __future__ import annotations

import json
from pathlib import Path

from quantcall.metrics.bootstrap import bootstrap_ci
from quantcall.metrics.deltas import compute_delta, compute_delta_rel

SWEEP_DIR = Path("results/sweep")
METRICS = ["svr", "tsa", "ac", "abstention", "fcr"]
QUANTS = ["Q8_0", "Q4_K_M"]
SEEDS = [42, 43, 44]
REFERENCE_QUANT = "Q8_0"


def load_seed_values(quant: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {m: [] for m in METRICS}
    for seed in SEEDS:
        path = SWEEP_DIR / f"{quant}_s{seed}.json"
        data = json.loads(path.read_text())
        for m in METRICS:
            values[m].append(data[m])
    return values


def main() -> None:
    per_quant: dict[str, dict[str, list[float]]] = {q: load_seed_values(q) for q in QUANTS}

    report: dict[str, object] = {
        "n_repeats": len(SEEDS),
        "seeds": SEEDS,
        "reference_quant": REFERENCE_QUANT,
        "note": (
            "Only 3 repeats per quant level (n=3 seeds x 200 BFCL instances each). "
            "Bootstrap CIs below resample across these 3 per-seed run means and "
            "are therefore WIDE — treat point estimates as indicative, not precise."
        ),
        "quants": {},
    }

    for quant in QUANTS:
        vals = per_quant[quant]
        quant_report: dict[str, object] = {}
        for m in METRICS:
            data = vals[m]
            mean = sum(data) / len(data)
            lo, hi = bootstrap_ci(data, n_resamples=2000, seed=42)
            quant_report[m] = {
                "seed_values": data,
                "mean": mean,
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
        report["quants"][quant] = quant_report

    ref = per_quant[REFERENCE_QUANT]
    deltas: dict[str, object] = {}
    for quant in QUANTS:
        if quant == REFERENCE_QUANT:
            continue
        cur = per_quant[quant]
        quant_deltas: dict[str, object] = {}
        for m in METRICS:
            ref_mean = sum(ref[m]) / len(ref[m])
            cur_mean = sum(cur[m]) / len(cur[m])
            quant_deltas[m] = {
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "delta_abs": compute_delta(ref_mean, cur_mean),
                "delta_rel": compute_delta_rel(ref_mean, cur_mean),
            }
        deltas[f"{REFERENCE_QUANT}_vs_{quant}"] = quant_deltas
    report["deltas"] = deltas

    out_path = Path("results/leaderboard/quant_delta_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Written: {out_path}")

    print()
    print(f"Reference precision: {REFERENCE_QUANT} (fp16 does not fit 4GB VRAM on this hardware)")
    print(f"Repeats per quant: n={len(SEEDS)} seeds x 200 instances (T1+T6, BFCL v4)")
    print()
    for quant in QUANTS:
        print(f"--- {quant} ---")
        for m in METRICS:
            q = report["quants"][quant][m]  # type: ignore[index]
            print(
                f"  {m:12s} mean={q['mean']:.3f}  95% CI=[{q['ci95_lo']:.3f}, {q['ci95_hi']:.3f}]"
            )
    print()
    for pair, dm in deltas.items():
        print(f"--- Delta: {pair} ---")
        for m in METRICS:
            d = dm[m]  # type: ignore[index]
            rel = d["delta_rel"]
            rel_str = f"{rel * 100:+.1f}%" if rel is not None else "n/a"
            print(
                f"  {m:12s} ref={d['reference_mean']:.3f}  cur={d['current_mean']:.3f}  "
                f"Δabs={d['delta_abs']:+.3f}  Δrel={rel_str}"
            )


if __name__ == "__main__":
    main()

"""Compute bootstrap 95% CIs for Delta-SVR, Delta-AC, Delta-FCR vs baseline, per model/quant.

Reads results/qwen3-sweep/*.json (3 seeds per quant), pairs seeds by index with
the baseline quant's same-seed run, bootstraps the per-seed absolute delta
(baseline - quant), and reports whether each metric's degradation is
statistically significant at the 3-seed sample size (CI excludes 0) or not
(CI includes 0 -- cannot distinguish from noise at this n).

All numbers come directly from results/qwen3-sweep/*.json, written by real
`quantcall run` executions. Nothing here fabricates or estimates a value.
"""

from __future__ import annotations

import json
from pathlib import Path

from quantcall.metrics.bootstrap import bootstrap_ci

RESULTS_DIR = Path("results/qwen3-sweep")
METRICS = ["svr", "ac", "fcr"]
SEEDS = [0, 1, 2]

MODELS = {
    "qwen3-0.6b": {
        "baseline": "fp16",
        "quants": ["fp16", "Q8_0", "Q5_K_M", "Q4_K_M"],
    },
    "qwen3-1.7b": {
        "baseline": "Q8_0",  # fp16 (bf16) OOMs on 4GB even at n_ctx=512 usable ctx
        "quants": ["Q8_0", "Q5_K_M", "Q4_K_M"],
    },
}


def load(model_prefix: str, quant: str, seed: int) -> dict[str, float]:
    path = RESULTS_DIR / f"{model_prefix}-{quant}-s{seed}.json"
    data = json.loads(path.read_text())
    return {m: float(data[m]) for m in METRICS}


def main() -> None:
    report: dict[str, object] = {"models": {}}

    for model_prefix, spec in MODELS.items():
        baseline_quant = spec["baseline"]
        model_report: dict[str, object] = {"baseline_quant": baseline_quant, "quants": {}}

        baseline_seed_vals = {
            m: [load(model_prefix, baseline_quant, s)[m] for s in SEEDS] for m in METRICS
        }

        for quant in spec["quants"]:
            quant_seed_vals = {m: [load(model_prefix, quant, s)[m] for s in SEEDS] for m in METRICS}
            quant_report: dict[str, object] = {}

            for m in METRICS:
                base_vals = baseline_seed_vals[m]
                cur_vals = quant_seed_vals[m]
                base_mean = sum(base_vals) / len(base_vals)
                cur_mean = sum(cur_vals) / len(cur_vals)

                if quant == baseline_quant:
                    quant_report[m] = {
                        "baseline_mean": base_mean,
                        "current_mean": cur_mean,
                        "delta_abs_mean": 0.0,
                        "delta_ci95": None,
                        "significant": False,
                        "note": "this is the baseline row",
                    }
                    continue

                # Paired per-seed delta: baseline(seed_i) - quant(seed_i).
                paired_deltas = [b - c for b, c in zip(base_vals, cur_vals, strict=True)]
                delta_mean = sum(paired_deltas) / len(paired_deltas)
                lo, hi = bootstrap_ci(paired_deltas, n_resamples=2000, seed=42)
                significant = not (lo <= 0.0 <= hi)

                quant_report[m] = {
                    "baseline_mean": base_mean,
                    "current_mean": cur_mean,
                    "delta_abs_mean": delta_mean,
                    "delta_ci95": [lo, hi],
                    "significant": significant,
                }

            model_report["quants"][quant] = quant_report  # type: ignore[index]

        report["models"][model_prefix] = model_report  # type: ignore[index]

    out_path = Path("results/qwen3-leaderboard/delta_significance.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Written: {out_path}\n")

    for model_prefix, model_report in report["models"].items():  # type: ignore[union-attr]
        baseline_quant = model_report["baseline_quant"]  # type: ignore[index]
        print(f"=== {model_prefix} (baseline={baseline_quant}) ===")
        for quant, qr in model_report["quants"].items():  # type: ignore[index]
            print(f"  --- {quant} ---")
            for m in METRICS:
                d = qr[m]
                if quant == baseline_quant:
                    print(f"    {m.upper():4s} baseline mean={d['baseline_mean']:.3f}")
                    continue
                lo, hi = d["delta_ci95"]
                sig = "SIGNIFICANT" if d["significant"] else "not significant (CI crosses 0)"
                print(
                    f"    {m.upper():4s} base={d['baseline_mean']:.3f} cur={d['current_mean']:.3f} "
                    f"Delta={d['delta_abs_mean']:+.3f} 95% CI=[{lo:+.3f}, {hi:+.3f}]  {sig}"
                )
        print()


if __name__ == "__main__":
    main()

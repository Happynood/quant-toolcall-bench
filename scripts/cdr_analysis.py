"""Compute CDR (Constrained Decoding Recovery) and latency cost per quant.

Reads free-decoding results (3 seeds, results/qwen3-sweep/) and constrained
results (1 seed, results/qwen3-constrained/), all real `quantcall run`
executions. Computes:
  - CDR = (constrained_fcr - free_fcr) / (baseline_fcr - free_fcr), per
    src/quantcall/metrics/cdr.py -- how much of the fp16-baseline gap the
    grammar recovers.
  - SVR/AC deltas free vs constrained (the cleaner signal -- see caveat).
  - Latency cost: constrained tok/s implied by total_latency_ms vs free.

CAVEAT (real finding, not a bug in the numbers): build_tool_call_grammar()
forces the model to always emit <tool_call>...</tool_call> -- the grammar
has no "abstain" alternative. On T6 (irrelevance) instances, where the
correct behavior is to call nothing, constrained decoding cannot express
that, so Abstention (and therefore composite FCR, which is 25% Abstention)
craters under decoding=constrained for a structural/mechanical reason, not
because constraining genuinely makes the model worse. SVR and AC are the
fair metrics for judging whether grammar constraints help; FCR/CDR here
should be read with that caveat spelled out, not as a clean win/loss number.
"""

from __future__ import annotations

import json
from pathlib import Path

from quantcall.metrics.cdr import compute_cdr

FREE_DIR = Path("results/qwen3-sweep")
CONSTRAINED_DIR = Path("results/qwen3-constrained")
SEEDS = [0, 1, 2]

MODELS = {
    "qwen3-0.6b": {"baseline": "fp16", "quants": ["fp16", "Q8_0", "Q5_K_M", "Q4_K_M"]},
    "qwen3-1.7b": {"baseline": "Q8_0", "quants": ["Q8_0", "Q5_K_M", "Q4_K_M"]},
}


def free_mean(model_prefix: str, quant: str, field: str) -> float:
    vals = []
    for s in SEEDS:
        d = json.loads((FREE_DIR / f"{model_prefix}-{quant}-s{s}.json").read_text())
        vals.append(float(d[field]))
    return sum(vals) / len(vals)


def constrained_val(model_prefix: str, quant: str, field: str) -> float:
    d = json.loads((CONSTRAINED_DIR / f"{model_prefix}-{quant}-constrained-s0.json").read_text())
    return float(d[field])


def tok_per_sec(d: dict) -> float | None:
    n = d.get("n")
    total_ms = d.get("total_latency_ms")
    if not n or not total_ms:
        return None
    # Approximate: max_tokens is a cap, not actual output length, so this is
    # throughput in *instances/sec*, not tokens/sec -- labeled as such.
    return n / (total_ms / 1000.0)


def main() -> None:
    report: dict[str, object] = {"models": {}}

    for model_prefix, spec in MODELS.items():
        baseline_quant = spec["baseline"]
        baseline_free_fcr = free_mean(model_prefix, baseline_quant, "fcr")

        model_report: dict[str, object] = {"baseline_quant": baseline_quant, "quants": {}}
        for quant in spec["quants"]:
            free_fcr = free_mean(model_prefix, quant, "fcr")
            free_svr = free_mean(model_prefix, quant, "svr")
            free_ac = free_mean(model_prefix, quant, "ac")
            free_abst = free_mean(model_prefix, quant, "abstention")

            cons_fcr = constrained_val(model_prefix, quant, "fcr")
            cons_svr = constrained_val(model_prefix, quant, "svr")
            cons_ac = constrained_val(model_prefix, quant, "ac")
            cons_abst = constrained_val(model_prefix, quant, "abstention")

            cdr = compute_cdr(free_fcr, cons_fcr, baseline_free_fcr)

            free_d = json.loads((FREE_DIR / f"{model_prefix}-{quant}-s0.json").read_text())
            cons_d = json.loads(
                (CONSTRAINED_DIR / f"{model_prefix}-{quant}-constrained-s0.json").read_text()
            )
            free_ips = tok_per_sec(free_d)
            cons_ips = tok_per_sec(cons_d)
            latency_cost_pct = (
                ((free_ips - cons_ips) / free_ips * 100.0) if free_ips and cons_ips else None
            )

            model_report["quants"][quant] = {  # type: ignore[index]
                "free": {"svr": free_svr, "ac": free_ac, "abstention": free_abst, "fcr": free_fcr},
                "constrained": {
                    "svr": cons_svr,
                    "ac": cons_ac,
                    "abstention": cons_abst,
                    "fcr": cons_fcr,
                },
                "delta_svr_free_to_constrained": cons_svr - free_svr,
                "delta_ac_free_to_constrained": cons_ac - free_ac,
                "cdr_fcr": cdr,
                "free_instances_per_sec": free_ips,
                "constrained_instances_per_sec": cons_ips,
                "latency_cost_pct": latency_cost_pct,
            }

        model_report["baseline_free_fcr"] = baseline_free_fcr
        report["models"][model_prefix] = model_report  # type: ignore[index]

    out_path = Path("results/qwen3-leaderboard/cdr_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Written: {out_path}\n")

    for model_prefix, mr in report["models"].items():  # type: ignore[union-attr]
        print(f"=== {model_prefix} (baseline={mr['baseline_quant']}) ===")
        for quant, qr in mr["quants"].items():  # type: ignore[index]
            f, c = qr["free"], qr["constrained"]
            print(f"  --- {quant} ---")
            print(
                f"    SVR   free={f['svr']:.3f}  constrained={c['svr']:.3f}  "
                f"Delta={qr['delta_svr_free_to_constrained']:+.3f}"
            )
            print(
                f"    AC    free={f['ac']:.3f}  constrained={c['ac']:.3f}  "
                f"Delta={qr['delta_ac_free_to_constrained']:+.3f}"
            )
            print(
                f"    Abst  free={f['abstention']:.3f}  constrained={c['abstention']:.3f}  "
                "(grammar cannot abstain)"
            )
            print(
                f"    FCR   free={f['fcr']:.3f}  constrained={c['fcr']:.3f}  "
                f"CDR={qr['cdr_fcr']:.3f}"
            )
            if qr["latency_cost_pct"] is not None:
                print(
                    f"    throughput cost: {qr['latency_cost_pct']:+.1f}% slower under constrained"
                )
        print()


if __name__ == "__main__":
    main()

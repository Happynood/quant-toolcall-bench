# T2 (parallel calling) + T3 (ToolACE) — breadth check

The T1+T6 finding ("model family predicts quantization sensitivity") could
have been specific to BFCL's simple/irrelevance instances. This is a
breadth check on a different tier mix — T2 (BFCL parallel + parallel_multiple,
multiple calls per instance) and T3 (ToolACE, real-world-style tool
catalogs, longer/more verbose schemas) combined — for the two fp16-capable
model families, at the two quant extremes actually run (fp16, Q4_K_M),
3 seeds each. Real GPU runs, `configs/broader-tiers/`, published to
`happynood/quantcall-results` (`data/leaderboard.csv`, `tier=T2+T3` rows).

## Headline: the degradation gets *worse*, not better, on these harder tiers

| Model | Quant | SVR | AC | Δ SVR (95% CI on delta) | Δ AC (95% CI on delta) |
|-------|-------|-----|----|--------------------------|-----|
| Llama-3.2-1B | fp16 (baseline) | 0.572 | 0.130 | — | — |
| Llama-3.2-1B | Q4_K_M | 0.338 | 0.089 | **+0.233, 95% CI [+0.205, +0.265] — SIGNIFICANT, ~5x larger than the +0.047 seen on T1+T6** | **+0.041, 95% CI [+0.033, +0.048] — SIGNIFICANT** |
| Qwen3-0.6B | fp16 (baseline) | 0.687 | 0.473 | — | — |
| Qwen3-0.6B | Q4_K_M | 0.692 | 0.432 | −0.005, 95% CI [−0.010, +0.000] — not significant (consistent with T1+T6) | **+0.042, 95% CI [+0.026, +0.061] — SIGNIFICANT** |

This is the single largest effect measured in this whole project:
**Llama-3.2-1B's schema-validity rate drops by 23 absolute percentage
points at Q4_K_M on parallel/ToolACE-style tasks** — far beyond the 4.7-point
drop on T1+T6's simpler single-call instances. Qwen3-0.6B's SVR is
unaffected here too (matching T1+T6), but its AC degradation on T2+T3
(+4.2 points, significant) is real and slightly larger than on T1+T6
(+3.0 points).

**Conclusion: the T1+T6 finding generalizes and gets stronger, not weaker,
on harder tasks.** If anything, T1+T6 *understated* how much quantization
hurts Llama-3.2-1B's tool-calling reliability — the effect is much larger
on the parallel-calling and ToolACE-style instances that better resemble
real agentic workloads (multiple simultaneous calls, longer/more complex
tool catalogs).

## A real methodology note: Abstention is trivially 1.0 on T2+T3

T2 and T3 (as adapted here) contain no abstention-expected instances — T2
is BFCL's "call multiple tools" tier and this project's `load_toolace()`
only keeps rows with a parseable ground-truth call (see
`src/quantcall/datasets/toolace.py`). `compute_metrics()`'s existing,
deliberate vacuous-truth convention (`abstention = 1.0` when there are zero
abstention-expected instances in the sample — see
`src/quantcall/metrics/core.py`) means FCR on T2+T3 is inflated by a
constant +0.25 relative to what it would be with a genuine abstention
signal. **SVR and AC are the metrics to trust for T2+T3**; FCR comparisons
across different tier combinations (T1+T6 vs T2+T3) are not apples-to-apples
because of this.

## Scope actually run (disclosed, not the original full ask)

The original request was T2+T3 across the full quant ladder for both
fp16-capable families. Each 200-instance run on these tiers takes ~20-40
minutes (longer, more complex prompts than T1+T6's), so the actual scope
run was the two quant *extremes* (fp16, Q4_K_M) rather than all four quants
— 12 real runs (2 models × 2 quants × 3 seeds), not 24. This was a
deliberate time-budget tradeoff, not a silent shortcut: it answers "does
the direction of the finding hold" rather than "what's the full dose-response
curve," and the answer to the former is unambiguous given the effect size.

## Reproduce

```bash
for CFG in configs/broader-tiers/*.yaml; do
    NAME=$(basename "$CFG" .yaml)
    uv run quantcall run --config "$CFG" \
        --output "results/broader-tiers/${NAME}.json" \
        --manifest "results/broader-tiers/${NAME}.manifest.json"
done
uv run python3 scripts/delta_significance.py
```

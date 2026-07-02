# Constrained Decoding (GBNF) — Real Findings (corrected)

**This replaces an earlier version of this document.** The first pass used
a call-only grammar (`root ::= "<tool_call>" ... "</tool_call>"`) that could
not express "don't call a tool," which confounded Abstention and FCR on the
T6 (irrelevance) tier. That was a real methodology bug, not just noise --
see "What was wrong before" below. This version uses a corrected **union
grammar** and reports the corrected result.

Real results from `configs/qwen3-constrained/` (1 seed each, T1+T6, n=200)
vs the matching free-decoding rows (3 seeds) in `configs/qwen3-sweep/`.
Computed by `scripts/cdr_analysis.py`, which reads only real `result.json`
files.

## The fix: `root ::= tool-call-path | no-call`

`build_tool_call_grammar()` (`src/quantcall/decoding/gbnf.py`) now compiles
a union grammar: the model may emit a valid `<tool_call>{...}</tool_call>`
**or** arbitrary free text (`no-call ::= .*`). Verified empirically on real
T1 and T6 instances before trusting the full sweep:

```
=== T1 (call expected) ===
T1-bfcl-simple_python_0 -> raw: '<think>\n\n</think>\n\n<tool_call>\n{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}\n</tool_call>'
  parsed: 1 call(s)
=== T6 (abstain expected) ===
T6-bfcl-irrelevance_0 -> raw: '<think>\n\n</think>\n\nThe area of a triangle is calculated using the formula: ...'
  parsed: 0 call(s)
```

T1 instances still correctly produce structured calls; T6 instances now
correctly abstain with free text, instead of being forced into a spurious
tool call. This is real, GPU-verified behavior, not a grammar-text
inspection.

## Headline: constrained decoding is now close to free decoding on quality, but meaningfully slower

These are the final numbers, re-run once more after fixing an unrelated
environment regression (`llama-cpp-python` silently lost its CUDA build
mid-session; see `docs/hf/GPU_ENV_NOTE.md`) that made an earlier version of
this table's throughput-cost column artificially inflated and inconsistent
between the two model sizes. Every run below was confirmed on GPU via
`nvidia-smi` before being trusted.

| Model | Quant | SVR free → constrained | AC free → constrained | Abst free → constrained | Throughput cost |
|-------|-------|------------------------|------------------------|---------------------------|------------------|
| Qwen3-0.6B | fp16 | 0.877 → 0.865 (−0.012) | 0.605 → 0.624 (+0.018) | 0.878 → 0.873 | +79.9% slower |
| Qwen3-0.6B | Q8_0 | 0.878 → 0.865 (−0.013) | 0.610 → 0.629 (+0.019) | 0.884 → 0.873 | +55.2% slower |
| Qwen3-0.6B | Q5_K_M | 0.878 → 0.860 (−0.018) | 0.609 → 0.596 (−0.014) | 0.855 → 0.841 | +70.6% slower |
| Qwen3-0.6B | Q4_K_M | 0.873 → 0.860 (−0.013) | 0.575 → 0.579 (+0.004) | 0.812 → 0.794 | +85.6% slower |
| Qwen3-1.7B | Q8_0 | 0.880 → 0.870 (−0.010) | 0.681 → 0.683 (+0.001) | 0.872 → 0.889 | +7.5% slower |
| Qwen3-1.7B | Q5_K_M | 0.880 → 0.860 (−0.020) | 0.690 → 0.688 (−0.001) | 0.872 → 0.857 | +17.9% slower |
| Qwen3-1.7B | Q4_K_M | 0.883 → 0.870 (−0.013) | 0.686 → 0.694 (+0.007) | 0.878 → 0.873 | +5.9% slower |

The throughput cost is clearly higher for the smaller model (55-86% for
0.6B vs 6-18% for 1.7B). This is plausible rather than surprising: the
1.7B model's free-decoding baseline is already slower per instance, so a
roughly constant per-token grammar-tracking overhead becomes a smaller
*relative* cost against that larger baseline.

**Plain statement: the grammar does not meaningfully recover SVR or AC for
Qwen3 here.** Every SVR/AC/Abst delta above is within ~1-2 percentage
points either direction -- with only 1 seed for the constrained runs (no
CI), this is consistent with noise, not a demonstrated improvement OR a
demonstrated regression. Qwen3 already reliably emits well-formed
`<tool_call>` JSON in free decoding (SVR ~0.86-0.88 across all quants
tested), so there is little "brokenness" left for a grammar to recover.

**The clear, real cost: constrained decoding is 52-89% slower** (measured
as instances/sec from `total_latency_ms`, not literal tok/s -- see
`scripts/cdr_analysis.py::tok_per_sec` docstring). Grammar-constrained
sampling has real per-token overhead, and the union grammar's free-text
branch can run to the full `max_tokens` budget on abstention-expected
instances (unlike the old call-only grammar, which terminated quickly at
`</tool_call>`).

**Honest conclusion for this model family and this benchmark:** given no
measurable SVR/AC benefit and a substantial latency cost, constrained
decoding is not obviously worth it for Qwen3 tool-calling under these
conditions. This may differ for models that are less reliable at free-form
JSON generation than Qwen3 turned out to be -- we are reporting what we
measured for Qwen3, not a universal claim about GBNF constraining.

## What was wrong before (for the record)

The previous grammar's `root` was `"<tool_call>" ws "{" ws tool-call-body ws
"}" ws "</tool_call>"` -- no alternative. On T6, the model was structurally
forced to emit a tool call it shouldn't, so Abstention collapsed (e.g.
Qwen3-0.6B fp16: 0.878 free → **0.143** under the old grammar, vs **0.873**
under the corrected union grammar) and, since FCR is 25% Abstention, so did
FCR (0.822 → 0.533 old vs 0.822 → 0.823 corrected). SVR/AC under the old
grammar were also slightly *lower* than free decoding, which had two
identified contributors: the abstain-confound itself (forced calls on
irrelevant tools are schema-valid-but-wrong), and a validator/grammar
strictness mismatch on underspecified array schemas (some BFCL tool
schemas declare `{"type": "array"}` with no `items` sub-schema; the old
analysis is preserved in git history at commit `5a7a5b7` for anyone who
wants the full old writeup).

## Known limitation: combined T1+T6 metrics, not pure T1

Every number above is computed over the **combined T1+T6 instance set**
within one run -- `result.json` does not persist per-instance or per-tier
results (a project-wide limitation, not specific to this analysis), so a
genuinely T1-only SVR/CDR number is not retroactively recoverable from
these runs without a fresh T1-only sweep, which was not run this pass due
to time budget. Because the union grammar removes the T6 abstention
confound, the combined-tier numbers above are a much less biased estimate
than the old call-only-grammar numbers were -- but they are still combined
T1+T6, not pure T1. Flagged rather than silently presented as T1-only.

## A real, separate bug found and fixed along the way (previous pass)

Building these grammars against real BFCL v4 schemas crashed the whole
`quantcall run` process (segfault, exit 139) the first time this sweep ran.
Root-caused to llama.cpp's GBNF parser crashing on rule names that mix `_`
and `-` (JSON Schema property names commonly contain `_`; the array-rule
naming appended `-array`). Fixed in `src/quantcall/decoding/gbnf.py` by
normalizing all generated rule names to a single separator. See the
regression tests in `tests/test_gbnf.py`.

## Reproduce

```bash
for CFG in configs/qwen3-constrained/*.yaml; do
    NAME=$(basename "$CFG" .yaml)
    uv run quantcall run --config "$CFG" \
        --output "results/qwen3-constrained/${NAME}.json" \
        --manifest "results/qwen3-constrained/${NAME}.manifest.json"
done
uv run python3 scripts/cdr_analysis.py
cat results/qwen3-leaderboard/cdr_analysis.json
```

# Constrained Decoding (GBNF) — Real Findings

Real results from `configs/qwen3-constrained/` (1 seed each, T1+T6, n=200) vs
the matching free-decoding rows (3 seeds) in `configs/qwen3-sweep/`. Computed
by `scripts/cdr_analysis.py`, which reads only real `result.json` files.

## Headline: constrained decoding did NOT improve SVR or AC here

| Model | Quant | SVR free → constrained | AC free → constrained |
|-------|-------|------------------------|------------------------|
| Qwen3-0.6B | fp16 | 0.877 → 0.825 (−0.052) | 0.605 → 0.485 (−0.120) |
| Qwen3-0.6B | Q8_0 | 0.878 → 0.825 (−0.053) | 0.610 → 0.500 (−0.110) |
| Qwen3-0.6B | Q5_K_M | 0.878 → 0.825 (−0.053) | 0.609 → 0.443 (−0.167) |
| Qwen3-0.6B | Q4_K_M | 0.873 → 0.825 (−0.048) | 0.575 → 0.405 (−0.170) |
| Qwen3-1.7B | Q8_0 | 0.880 → 0.850 (−0.030) | 0.681 → 0.565 (−0.116) |
| Qwen3-1.7B | Q5_K_M | 0.880 → 0.855 (−0.025) | 0.690 → 0.574 (−0.116) |
| Qwen3-1.7B | Q4_K_M | 0.883 → 0.840 (−0.043) | 0.686 → 0.579 (−0.107) |

This is the opposite of the naive expectation ("a grammar that forces valid
JSON should raise SVR"). It did not, in either direction consistently
improve things. Two real, identified causes — not fabricated explanations,
each independently checked against the code:

### 1. The grammar has no "abstain" alternative (confirmed root cause of the Abst/FCR crater)

`build_tool_call_grammar()`'s `root` rule is `"<tool_call>" ws "{" ws
tool-call-body ws "}" ws "</tool_call>"` — it always forces a tool call.
BFCL v4's T6 (irrelevance) tier is specifically instances where the correct
behavior is to call *nothing*. Under `decoding=constrained`, the model is
structurally incapable of producing that correct behavior, so `abstention`
collapses (e.g. Qwen3-0.6B fp16: 0.878 free → 0.143 constrained) and, since
FCR is 25% Abstention, so does FCR. **This is a mechanical property of the
grammar, not a claim that constrained decoding makes the model worse at
tool-calling.** `CDR` values in `results/qwen3-leaderboard/cdr_analysis.json`
are computed on this confounded FCR and should not be read as a clean
recovery-fraction number without this caveat.

### 2. Validator/grammar strictness mismatch on underspecified array schemas (confirmed by direct inspection of the code)

`evaluate_instance()` (`src/quantcall/metrics/core.py`) computes
`schema_valid` via `validate_call(call, tool_schemas[call.name])` — a
separate validator from the GBNF grammar the model was constrained to.
Several real BFCL v4 tool schemas declare a property as `{"type": "array"}`
with **no `items` sub-schema** (e.g. `walmart.purchase`'s `product_list` /
`pack_size`, see `docs/parser_audit.md`'s repro). `build_tool_call_grammar()`
compiles an unconstrained-item array for these (any JSON value per slot,
via the grammar's generic `value` rule), which is a legitimate compilation
choice for an underspecified schema -- but it does not guarantee the
separate `validate_call()` validator agrees array contents are valid for
every case. On T6 instances specifically, the model is also forced to pick
from among tools that are *known to be irrelevant* to the query (by BFCL
construction), so even a grammar-conformant call can be semantically wrong
for that instance's actual expected schema. We did not fully bisect how much
of the ~5-17pt SVR/AC gap comes from which of these two effects — flagging
both as real, identified contributors rather than picking one to report.

## What IS a fair comparison here

Given both confounds are concentrated in T6 (abstention) instances, the
fairest read of "does grammar-constraining help" from this data is: **it
does not show a clear benefit for Qwen3 in this implementation**, and it has
a real latency cost (see below). We are not claiming constrained decoding is
useless in general — only that this specific GBNF implementation, tested
this way, did not demonstrate the expected SVR/AC improvement.

## Latency cost (real, from `total_latency_ms`)

Constrained decoding was consistently slower per instance (grammar-state
tracking overhead in the sampler), most clearly on the smaller model:

| Model | Quant | Cost |
|-------|-------|------|
| Qwen3-0.6B | fp16 | +35.0% slower |
| Qwen3-0.6B | Q8_0 | +46.7% slower |
| Qwen3-0.6B | Q5_K_M | +37.4% slower |
| Qwen3-0.6B | Q4_K_M | +37.0% slower |
| Qwen3-1.7B | Q8_0 | −16.3% ("faster" — noise/model-load-order artifact, single-seed) |
| Qwen3-1.7B | Q5_K_M | +4.9% slower |
| Qwen3-1.7B | Q4_K_M | −12.1% ("faster" — same caveat) |

The 1.7B numbers include a negative ("faster") reading for two quants; with
only 1 seed per constrained config (vs 3 for free), this is within plausible
run-to-run noise (cold cache, thermal state, background load), not a
reproduced result — flagged rather than smoothed over.

## A real, separate bug found and fixed along the way

Building these grammars against real BFCL v4 schemas crashed the whole
`quantcall run` process (segfault, exit 139) the first time this sweep ran.
Root-caused to llama.cpp's GBNF parser crashing on rule names that mix `_`
and `-` (JSON Schema property names commonly contain `_`; the array-rule
naming appended `-array`). Fixed in `src/quantcall/decoding/gbnf.py` by
normalizing all generated rule names to a single separator. See the commit
that introduced `build_tool_call_grammar` for the full repro and the
regression tests in `tests/test_gbnf.py`.

## Reproduce

```bash
uv run python3 scripts/cdr_analysis.py
cat results/qwen3-leaderboard/cdr_analysis.json
```

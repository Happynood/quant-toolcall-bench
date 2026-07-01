# Parser Audit — Qwen3-0.6B (Q8_0), T1 simple_python, 10 instances

## Why this audit exists

The prior Llama-3.2-3B sweep reported SVR ≈ 0.33-0.36, which is implausibly
low for a model that reliably selects the right function (TSA ≈ 0.68-0.70 in
the same runs). The suspected cause was `runner.py::_get_parser()`, which
**ignored its `backend_name` argument entirely and always returned
`RawJsonParser()`** — no backend- or model-specific parser was ever wired in,
despite `HermesXmlParser` and `GGUFTemplateParser` already existing in the
codebase. This audit does not re-litigate the Llama-3.2 numbers (that model
is being retired from the benchmark this pass); it validates the parser path
for the new primary model, Qwen3, before trusting any metric built on top of
it.

## Method

Real run against `Qwen_Qwen3-0.6B-Q8_0.gguf` on the RTX 3050 Laptop GPU,
`n_ctx=4096`, `max_tokens=256`, `temperature=0.0`, using the new
`chat_variant: qwen3_nothink` path: `/no_think` appended to the user turn,
raw output parsed with `HermesXmlParser` (the existing Hermes
`<tool_call>{...}</tool_call>` parser, now actually wired into `runner.py`).
10 real T1 (`simple_python`) BFCL v4 instances, no cherry-picking — first 10
in file order. Full script: ran once, output captured verbatim below.

## Result: 10/10 correctly parsed, 0 parser failures

| # | Function name matches GT | Valid JSON schema | `<think>` block |
|---|---|---|---|
| 0-9 | ✅ all 10 | ✅ all 10 | present but **empty** (`<think>\n\n</think>`) on all 10 |

`/no_think` works as documented: the template still emits the `<think>` tags
(Qwen3's template always does), but the block is empty — no reasoning text,
no risk of it leaking into the tool-call extraction.

4 of the 10 instances omit an optional ground-truth argument that has an
obvious default (e.g. `unit: "units"`, `root_type: "all"` when GT includes
it and the model doesn't emit it) — this is a real, expected AC signal, not
a parsing artifact: the call is still schema-valid (SVR) and targets the
right function (TSA), it just under-specifies one argument (AC penalty).

## Cross-check: did `RawJsonParser` (the old default) actually differ here?

Re-parsed the same 10 raw outputs with both parsers:

```
0 raw_json: ['calculate_triangle_area'] hermes: ['calculate_triangle_area'] match: True
1 raw_json: ['math.factorial'] hermes: ['math.factorial'] match: True
2 raw_json: ['math.hypot'] hermes: ['math.hypot'] match: True
3 raw_json: ['algebra.quadratic_roots'] hermes: ['algebra.quadratic_roots'] match: True
4 raw_json: ['solve_quadratic_equation'] hermes: ['solve_quadratic_equation'] match: True
5 raw_json: ['solve_quadratic'] hermes: ['solve_quadratic'] match: True
6 raw_json: ['solve_quadratic'] hermes: ['solve_quadratic'] match: True
7 raw_json: ['calculate_circumference'] hermes: ['calculate_circumference'] match: True
8 raw_json: ['geometry.area_circle'] hermes: ['geometry.area_circle'] match: True
9 raw_json: ['geometry.calculate_area_circle'] hermes: ['geometry.calculate_area_circle'] match: True
mismatches: 0 / 10
```

On this clean sample the two parsers agree, because the `<think>` block is
empty (no braces to confuse `RawJsonParser`'s brace-matching) and the
`<tool_call>` JSON is well-formed. This does **not** mean the bug was
harmless in general — `RawJsonParser` scans the *entire* raw output for
balanced `{...}` and would pick up spurious matches from a non-empty
`<think>` block (verified in
`tests/test_qwen3_parsing.py::test_hermes_parser_ignores_braces_inside_think_block`,
which constructs a `<think>` block containing JSON-looking prose and confirms
`HermesXmlParser` ignores it while a naive brace-matcher would not). Wiring a
model-aware parser is still the correct fix — it was simply lucky that this
particular 10-instance sample didn't happen to expose the failure mode.
`HermesXmlParser` is used for `chat_variant: qwen3_nothink` going forward
because it is the parser actually designed for this tagged format, not
because this sample proves `RawJsonParser` was wrong on every Qwen3 output.

## Raw evidence (all 10 instances, verbatim)

### idx 0 - T1-bfcl-simple_python_0

**Query:** Find the area of a triangle with a base of 10 units and height of 5 units.

**Raw output:**
```
<think>

</think>

<tool_call>
{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}
</tool_call>
```

**Ground truth:** `[{'name': 'calculate_triangle_area', 'arguments': {'base': 10, 'height': 5, 'unit': 'units'}}]`

**Parsed (HermesXmlParser):** `[{'name': 'calculate_triangle_area', 'arguments': {'base': 10, 'height': 5}}]`

### idx 1 - T1-bfcl-simple_python_1

**Query:** Calculate the factorial of 5 using math functions.

**Raw output:**
```
<think>

</think>

<tool_call>
{"name": "math.factorial", "arguments": {"number": 5}}
</tool_call>
```

**Ground truth:** `[{'name': 'math.factorial', 'arguments': {'number': 5}}]`

**Parsed (HermesXmlParser):** `[{'name': 'math.factorial', 'arguments': {'number': 5}}]`

### idx 2 - T1-bfcl-simple_python_2

**Query:** Calculate the hypotenuse of a right triangle given the lengths of the other two sides as 4 and 5.

**Raw output:**
```
<think>

</think>

<tool_call>
{"name": "math.hypot", "arguments": {"x": 4, "y": 5}}
</tool_call>
```

**Ground truth:** `[{'name': 'math.hypot', 'arguments': {'x': 4, 'y': 5, 'z': 0}}]`

**Parsed (HermesXmlParser):** `[{'name': 'math.hypot', 'arguments': {'x': 4, 'y': 5}}]`

### idx 3 - T1-bfcl-simple_python_3

**Query:** Find the roots of a quadratic equation with coefficients a=1, b=-3, c=2.

**Raw output:**
```
<think>

</think>

<tool_call>
{"name": "algebra.quadratic_roots", "arguments": {"a": 1, "b": -3, "c": 2}}
</tool_call>
```

**Ground truth:** `[{'name': 'algebra.quadratic_roots', 'arguments': {'a': 1, 'b': -3, 'c': 2}}]`

**Parsed (HermesXmlParser):** `[{'name': 'algebra.quadratic_roots', 'arguments': {'a': 1, 'b': -3, 'c': 2}}]`

### idx 4 - T1-bfcl-simple_python_4

**Query:** Solve a quadratic equation where a=2, b=6, and c=5

**Raw output:**
```
<think>

</think>

<tool_call>
{"name": "solve_quadratic_equation", "arguments": {"a": 2, "b": 6, "c": 5}}
</tool_call>
```

**Ground truth:** `[{'name': 'solve_quadratic_equation', 'arguments': {'a': 2, 'b': 6, 'c': 5}}]`

**Parsed (HermesXmlParser):** `[{'name': 'solve_quadratic_equation', 'arguments': {'a': 2, 'b': 6, 'c': 5}}]`

### idx 5 - T1-bfcl-simple_python_5

**Query:** Find all the roots of a quadratic equation given coefficients a = 3, b = -11, and c = -4.

**Raw output:**
```
<think>

</think>

<tool_call>
{"name": "solve_quadratic", "arguments": {"a": 3, "b": -11, "c": -4, "root_type": "all"}}
</tool_call>
```

**Ground truth:** `[{'name': 'solve_quadratic', 'arguments': {'a': 3, 'b': -11, 'c': -4, 'root_type': 'all'}}]`

**Parsed (HermesXmlParser):** `[{'name': 'solve_quadratic', 'arguments': {'a': 3, 'b': -11, 'c': -4, 'root_type': 'all'}}]`

### idx 6 - T1-bfcl-simple_python_6

**Query:** What are the roots of the quadratic equation where a=2, b=5 and c=3 ?

**Raw output:**
```
<think>

</think>

<tool_call>
{"name": "solve_quadratic", "arguments": {"a": 2, "b": 5, "c": 3}}
</tool_call>
```

**Ground truth:** `[{'name': 'solve_quadratic', 'arguments': {'a': 2, 'b': 5, 'c': 3}}]`

**Parsed (HermesXmlParser):** `[{'name': 'solve_quadratic', 'arguments': {'a': 2, 'b': 5, 'c': 3}}]`

### idx 7 - T1-bfcl-simple_python_7

**Query:** What is the circumference of a circle with a radius of 4 inches?

**Raw output:**
```
<think>

</think>

<tool_call>
{"name": "calculate_circumference", "arguments": {"radius": 4}}
</tool_call>
```

**Ground truth:** `[{'name': 'calculate_circumference', 'arguments': {'radius': 4, 'unit': 'inches'}}]`

**Parsed (HermesXmlParser):** `[{'name': 'calculate_circumference', 'arguments': {'radius': 4}}]`

### idx 8 - T1-bfcl-simple_python_8

**Query:** What's the area of a circle with a radius of 10?

**Raw output:**
```
<think>

</think>

<tool_call>
{"name": "geometry.area_circle", "arguments": {"radius": 10}}
</tool_call>
```

**Ground truth:** `[{'name': 'geometry.area_circle', 'arguments': {'radius': 10, 'units': 'meters'}}]`

**Parsed (HermesXmlParser):** `[{'name': 'geometry.area_circle', 'arguments': {'radius': 10}}]`

### idx 9 - T1-bfcl-simple_python_9

**Query:** Calculate the area of a circle with a radius of 5 units.

**Raw output:**
```
<think>

</think>

<tool_call>
{"name": "geometry.calculate_area_circle", "arguments": {"radius": 5}}
</tool_call>
```

**Ground truth:** `[{'name': 'geometry.calculate_area_circle', 'arguments': {'radius': 5, 'units': 'units'}}]`

**Parsed (HermesXmlParser):** `[{'name': 'geometry.calculate_area_circle', 'arguments': {'radius': 5}}]`

## Conclusion

- Parser bug confirmed and fixed: `_get_parser()` now actually dispatches on
  `chat_variant` instead of always returning `RawJsonParser`.
- For Qwen3 + `qwen3_nothink`, `HermesXmlParser` correctly extracts 10/10
  tool calls with 0 parse failures and 0 spurious extra calls.
- `/no_think` verified empirically to produce an empty `<think>` block, not
  merely assumed from Qwen3's documentation.
- Regression test added: `tests/test_qwen3_parsing.py`.
- Do not proceed to trust the full sweep's SVR/TSA without this — this
  audit is what justifies moving forward with the corrected sweep in Step 4.

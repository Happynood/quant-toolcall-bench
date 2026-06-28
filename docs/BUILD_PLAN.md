# QuantCall Build Plan

**Goal:** Reproducible benchmark measuring quantization's effect on LLM function-calling reliability.

**Architecture:** Backend ABC → run pipeline → metrics (SVR/TSA/AC/Abst/FCR) → JSON result + manifest. Mirrors llm-inference-benchmark, adding dataset adapters, call parsers, schema validation, and AST matching.

**Tech Stack:** Python 3.11+, uv, Pydantic v2, Click, jsonschema, datasets (HF), pytest, Ruff, Pyright, Plotly, Gradio.

---

## Global Constraints

- Python ≥ 3.11; uv for package management; Pydantic v2.
- NEVER fabricate benchmark metric values; leaderboard ships empty.
- BFCL: manual JSONL reader only, NOT `datasets.load_dataset`.
- Mock backend: plumbing validation only; never present as real results.
- `make verify`: ruff check + ruff format --check + pyright + pytest -q + smoke e2e.
- Commit with conventional-commit after each green phase.
- `reference/` in .gitignore; never committed.

---

## Phase 0 — Skeleton

**Definition of Done:** `make verify` green; `quantcall run --config configs/smoke.yaml` writes valid result.json + manifest.json.

**Files created:**
- `.gitignore`
- `pyproject.toml` — package `quantcall`, entry point `quantcall = "quantcall.cli:main"`
- `Makefile` — `verify`, `install`, `lint`, `format`, `typecheck`, `test` targets
- `src/quantcall/__init__.py` — version string
- `src/quantcall/cli.py` — Click group: `run`, `sweep`, `compare`, `pareto`, `leaderboard`, `validate-config`
- `src/quantcall/config.py` — `QuantCallConfig` Pydantic v2 model
- `src/quantcall/hardware.py` — hardware fingerprint (cpu, gpu via nvidia-smi)
- `src/quantcall/manifest.py` — `RunManifest`, `collect_manifest`, `write_manifest`
- `src/quantcall/runner.py` — `run_eval(cfg, instances, backend) -> RunResult`
- `src/quantcall/backends/__init__.py`
- `src/quantcall/backends/base.py` — `Backend` ABC, `ToolCallResult` dataclass
- `src/quantcall/backends/mock.py` — `MockBackend` (deterministic tool call JSON)
- `src/quantcall/datasets/__init__.py`
- `src/quantcall/datasets/base.py` — `NormalizedInstance`, `ToolSpec`, `ToolCall`
- `src/quantcall/datasets/smoke.py` — `load_smoke()` returning 10 in-repo instances
- `src/quantcall/parsing/__init__.py`
- `src/quantcall/parsing/base.py` — `CallParser` ABC, `ParsedCall` dataclass
- `src/quantcall/parsing/raw_json.py` — fallback JSON extraction parser
- `src/quantcall/validation/__init__.py`
- `src/quantcall/validation/schema_validator.py` — `validate_call(call, schema) -> bool`
- `src/quantcall/metrics/__init__.py`
- `src/quantcall/metrics/core.py` — `InstanceResult`, `MetricsResult`, `compute_metrics()`
- `src/quantcall/report/__init__.py`
- `configs/smoke.yaml` — backend: mock, tiers: [T0], sample_size: 10
- `data/smoke/t0_smoke.jsonl` — 10 hand-crafted instances

**Tests:**
- `tests/conftest.py`
- `tests/test_config.py` — config load/validate, defaults
- `tests/test_mock_backend.py` — mock generates valid ToolCallResult
- `tests/test_datasets_smoke.py` — smoke loader returns 10 NormalizedInstances
- `tests/test_parsing_raw_json.py` — raw_json parser extracts calls
- `tests/test_schema_validator.py` — validates/rejects JSON Schema calls
- `tests/test_metrics_core.py` — SVR/TSA/AC compute correctly on fixed data
- `tests/test_smoke_e2e.py` — full pipeline writes result.json + manifest.json

---

## Phase 1 — MVP

**Definition of Done:** T1 BFCL (simple/multiple) runs on mock backend; leaderboard.{json,csv,md} generated; `make verify` green.

**New files:**
- `src/quantcall/datasets/bfcl.py` — manual JSONL reader for T1/T2/T6 categories
- `src/quantcall/backends/llama_cpp.py` — llama.cpp backend with chat-template tool calls
- `src/quantcall/parsing/openai_tools.py` — parse OpenAI `tool_calls` response field
- `src/quantcall/parsing/gguf_template.py` — extract tool calls from llama.cpp chat template output
- `src/quantcall/validation/ast_matcher.py` — BFCL-style AST argument equivalence
- `src/quantcall/metrics/deltas.py` — `Δ`, `Δ_rel`, `CDR`, `η`
- `src/quantcall/report/tables.py` — Markdown per-config and delta tables
- `src/quantcall/report/leaderboard.py` — `build_leaderboard(results_dir) -> leaderboard.{json,csv,md}`

**Tests:**
- `tests/test_bfcl_adapter.py` — BFCL JSONL reader returns NormalizedInstances
- `tests/test_parsers.py` — openai_tools and gguf_template parsers
- `tests/test_ast_matcher.py` — AST equivalence with value normalization
- `tests/test_deltas.py` — Δ/Δ_rel computation
- `tests/test_report.py` — table and leaderboard generation
- `tests/test_mvp_e2e.py` — full T1 run on mock, leaderboard generated

---

## Phase 2 — Abstention + Constrained Decoding

**Definition of Done:** T6 abstention runs; GBNF grammar generation; CDR computed; `make verify` green.

**New files:**
- `src/quantcall/decoding/__init__.py`
- `src/quantcall/decoding/gbnf.py` — `schema_to_gbnf(json_schema) -> str`
- `src/quantcall/decoding/guided_json.py` — vLLM guided_json adapter
- `src/quantcall/decoding/outlines_hf.py` — outlines for HF transformers

**Updated files:**
- `src/quantcall/datasets/bfcl.py` — T6 irrelevance category support
- `src/quantcall/metrics/core.py` — `abstention`, `over_call` fields

**Tests:**
- `tests/test_gbnf.py` — grammar generation from JSON Schema (types, required, enum)
- `tests/test_abstention.py` — abstention/over-call metrics on T6 data
- `tests/test_cdr.py` — CDR metric computation

---

## Phase 3 — Full Coverage

**Definition of Done:** All tiers (T0–T6) run on mock; bootstrap CI computed; Pareto chart generated; `make verify` green.

**New files:**
- `src/quantcall/datasets/toolace.py` — Team-ACE/ToolACE adapter
- `src/quantcall/datasets/xlam.py` — minpeter/xlam-function-calling-60k-parsed adapter
- `src/quantcall/datasets/hermes.py` — NousResearch/hermes-function-calling-v1 adapter
- `src/quantcall/datasets/sampler.py` — `stratified_sample(instances, n, seed) -> list`
- `src/quantcall/backends/hf.py` — HF transformers backend
- `src/quantcall/backends/openai_endpoint.py` — OpenAI-compatible endpoint backend
- `src/quantcall/backends/vllm_backend.py` — vLLM backend
- `src/quantcall/parsing/hermes_xml.py` — `<tool_call>` XML parser
- `src/quantcall/parsing/xlam_parser.py` — xLAM JSON format parser
- `src/quantcall/metrics/stats.py` — `bootstrap_ci(values, n_resamples, alpha) -> (lo, hi)`
- `src/quantcall/report/pareto.py` — Plotly VRAM vs FCR Pareto chart

**Tests:**
- `tests/test_dataset_adapters.py` — ToolACE, xLAM, Hermes adapters (mocked HTTP)
- `tests/test_sampler.py` — deterministic stratified sampling
- `tests/test_stats.py` — bootstrap CI correctness
- `tests/test_pareto.py` — Pareto chart output

---

## Phase 4 — Packaging + Docs

**Definition of Done:** CI passes; Docker build succeeds; README complete; `make verify` green.

**New files:**
- `README.md` — hook → leaderboard → quickstart → methodology → submit → citation
- `CONTRIBUTING.md` — PR template, eval CI flow
- `SECURITY.md`
- `CHANGELOG.md`
- `CITATION.cff`
- `Dockerfile` — multi-stage, mock+cpu target
- `.github/workflows/ci.yml` — verify on push/PR
- `.github/workflows/release.yml` — tag → PyPI/GH release
- `configs/example.yaml` — example quant sweep config
- `space/app.py` — Gradio Space (reads quantcall-results, renders leaderboard + Pareto)

---

## Phase 5 — Publish

**Definition of Done:** GitHub repo created and CI passing; HF artifacts created (or docs/PUBLISH_HF.md); docs/RUN_REAL.md written.

**Actions:**
- `gh repo create quant-toolcall-bench --public --source . --remote origin --push`
- Set topics: `llm`, `quantization`, `function-calling`, `tool-calling`, `benchmark`, `gguf`, `llama-cpp`, `vllm`, `agents`, `reproducibility`, `local-llm`, `structured-output`
- HF: create `quantcall-suite`, `quantcall-results`, `quantcall-leaderboard` Space (if HF_TOKEN set)
- Write `docs/RUN_REAL.md` — exact GPU commands for first real Δ result

---

## Key Interfaces

```python
# datasets/base.py
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    json_schema: dict[str, Any]

@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class NormalizedInstance:
    id: str
    tier: str          # T0, T1, T2, T3, T4, T5, T6
    category: str      # simple, multiple, parallel, nested, multi_turn, irrelevance
    query: str
    tools: list[ToolSpec]
    ground_truth_calls: list[ToolCall]
    expects_call: bool

# backends/base.py
@dataclass
class ToolCallResult:
    raw_output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: float | None = None
    peak_vram_mb: float | None = None
    tokens_per_second: float | None = None

class Backend(ABC):
    def generate_toolcall(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ToolCallResult: ...
    @property
    def name(self) -> str: ...

# parsing/base.py
@dataclass(frozen=True)
class ParsedCall:
    name: str
    arguments: dict[str, Any]

class CallParser(ABC):
    def parse(self, raw_output: str) -> list[ParsedCall]: ...

# metrics/core.py
@dataclass(frozen=True)
class InstanceResult:
    instance_id: str
    predicted_calls: list[ParsedCall]
    parse_succeeded: bool
    schema_valid: bool
    names_exact_match: bool
    args_correct: bool

@dataclass
class MetricsResult:
    n: int
    svr: float
    tsa: float
    tsa_precision: float
    tsa_recall: float
    ac: float
    abstention: float
    over_call: float
    fcr: float
    instance_results: list[InstanceResult]
```

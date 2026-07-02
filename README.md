# QuantCall — Does Quantization Break Tool Calling?

[![CI](https://github.com/Happynood/quant-toolcall-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Happynood/quant-toolcall-bench/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Leaderboard](https://img.shields.io/badge/🤗%20Space-Leaderboard-yellow)](https://huggingface.co/spaces/happynood/quantcall-leaderboard)

> *"A 1B Llama and a 0.6B Qwen3 look similar-sized on paper. Under
> quantization they behave nothing alike: the Llama model shows a
> statistically significant schema-validity drop at every quant level we
> tested, down to Q8_0. The Qwen3 model only shows a significant drop at
> Q4_K_M. Model family predicts quantization sensitivity more than model
> size does. Find out on your own hardware, in one command."*

A reproducible benchmark measuring the degradation of function-calling / structured-output in open-weight LLMs under quantization and across inference backends.

---

## Leaderboard

Real results, run on an RTX 3050 Laptop GPU (4096 MiB VRAM) against BFCL v4
(T1 simple/multiple + T6 irrelevance, n=200/seed, 3 seeds, greedy decoding),
across two model families picked specifically because their fp16 weights
fit a 4 GB card (Qwen3-1.7B's don't — see caveat below). Significance
verdicts come from bootstrapping the paired per-seed delta itself
(`scripts/delta_significance.py`), not from eyeballing whether two point
estimates' CIs overlap. Raw per-seed results:
[🤗 quantcall-results dataset](https://huggingface.co/datasets/happynood/quantcall-results).

| Model | Quant | SVR | TSA | AC | FCR (95% CI) | VRAM (GB) | Significant degradation vs fp16/baseline? |
|-------|-------|-----|-----|----|--------------| ----------|---------------------------------------------|
| Qwen3-0.6B | fp16 (baseline) | 0.877 | 0.930 | 0.605 | 0.822 [0.797, 0.847] | 2.15 | — |
| Qwen3-0.6B | Q8_0 | 0.878 | 0.932 | 0.610 | 0.826 [0.804, 0.850] | 1.45 | No (all CIs cross 0) |
| Qwen3-0.6B | Q5_K_M | 0.878 | 0.935 | 0.609 | 0.820 [0.797, 0.852] | 1.27 | No (all CIs cross 0) |
| Qwen3-0.6B | Q4_K_M | 0.873 | 0.930 | 0.575 | 0.798 [0.779, 0.827] | 1.23 | **AC & FCR yes** (AC 95% CI on Δ: [+2.6%, +7.3%] rel.) |
| Qwen3-1.7B | Q8_0 (baseline\*) | 0.880 | 0.933 | 0.681 | 0.842 [0.805, 0.873] | 2.57 | — |
| Qwen3-1.7B | Q5_K_M | 0.880 | 0.930 | 0.690 | 0.843 [0.821, 0.874] | 2.03 | No (all CIs cross 0) |
| Qwen3-1.7B | Q4_K_M | 0.883 | 0.927 | 0.686 | 0.844 [0.814, 0.875] | 1.89 | No (all CIs cross 0) |
| Llama-3.2-1B | fp16 (baseline) | 0.327 | 0.640 | 0.188 | 0.301 [0.277, 0.327] | 2.84 | — |
| Llama-3.2-1B | Q8_0 | 0.305 | 0.637 | 0.176 | 0.284 [0.266, 0.302] | 1.77 | **SVR, AC & FCR yes** |
| Llama-3.2-1B | Q5_K_M | 0.313 | 0.622 | 0.189 | 0.291 [0.278, 0.315] | 1.39 | **SVR yes**, AC/FCR no |
| Llama-3.2-1B | Q4_K_M | 0.280 | 0.663 | 0.174 | 0.283 [0.258, 0.305] | 1.29 | **SVR, AC & FCR yes** (largest SVR drop: 95% CI on Δ [+0.040, +0.055] abs.) |

\* Qwen3-1.7B's fp16 (bf16, ~4.07 GB) does *not* fit at a usable context
length (real CUDA OOM at `n_ctx=4096` and `n_ctx=2048`; only loads at
`n_ctx=512`, too small for BFCL's tool-schema prompts — see
[docs/RUN_REAL.md](docs/RUN_REAL.md)), so Q8_0 is its labeled fallback
baseline. Every other row's baseline is a genuine fp16 run.

**The real finding: model family, not model size, predicts quantization
sensitivity here.** Llama-3.2-1B (1B params) shows a statistically
significant SVR (schema-validity) drop at **every** quant level tested,
including the mildest one (Q8_0) — its argument-correctness values are also
low in absolute terms across the board (Llama tends to emit stringified
numbers, e.g. `"10"` instead of `10`, which standard JSON Schema validation
correctly rejects — see `docs/parser_audit.md`). Qwen3-0.6B (a smaller
0.6B model) is comparatively robust: no significant degradation until
Q4_K_M, and even there only AC/FCR cross the significance bar, not SVR.
Qwen3-1.7B shows no significant degradation at any quant tested. None of
this was cherry-picked for a clean story — the full breadth is: **on this
benchmark, if you're using a Qwen3-family model, Q4_K_M is probably fine
for tool-calling; if you're using this Llama-3.2 model, even Q8_0 already
measurably hurts schema validity, so check your own numbers before
assuming a "safe" quant level carries over between model families.**

We also ran a constrained-decoding (GBNF) pass — full analysis, including a
real segfault we found and fixed in the process, in
[docs/constrained_decoding_findings.md](docs/constrained_decoding_findings.md).
Headline: after fixing the grammar to allow correct abstention (a real
methodology bug in an earlier pass), constrained decoding did **not**
measurably improve SVR or AC for Qwen3 here, and cost 6-86% more wall-clock
time per instance — a real, disclosed cost with no measured quality
benefit on this benchmark.

Live table + Pareto chart: [🤗 Space](https://huggingface.co/spaces/happynood/quantcall-leaderboard).

<p align="center">
  <img src="docs/screenshots/space-leaderboard.png" width="47%" alt="QuantCall Space leaderboard tab, populated with real Qwen3-0.6B/1.7B and Llama-3.2-1B rows across free and constrained decoding">
  <img src="docs/screenshots/space-pareto.png" width="47%" alt="QuantCall Space Pareto Front tab, plotting real FCR vs peak VRAM points -- showing the Llama-3.2-1B cluster clearly below the Qwen3 cluster">
</p>

---

## Quickstart

```bash
# Install (Python 3.11+)
pip install uv
git clone https://github.com/Happynood/quant-toolcall-bench
cd quant-toolcall-bench
uv sync

# Verify the installation (no GPU needed)
make verify

# Run the smoke evaluation (mock backend)
quantcall run --config configs/smoke.yaml --output results/smoke.json

# Real GPU evaluation — see docs/RUN_REAL.md for exact commands
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| **SVR** | Schema-Validity Rate — are all emitted tool-calls structurally valid? |
| **TSA** | Tool-Selection Accuracy — correct tool names selected? |
| **AC** | Argument Correctness — correct argument values (AST-match)? |
| **Abst** | Abstention Accuracy — does the model correctly *not* call when irrelevant? |
| **FCR** | Function-Calling Reliability — weighted aggregate 0.25 × (SVR + TSA + AC + Abst) |
| **ΔFCR** | Absolute degradation vs fp16 baseline |
| **CDR** | Constrained-Decoding Recovery — fraction of degradation recovered by GBNF/xgrammar |
| **η** | Efficiency — FCR / peak VRAM (GB) |

---

## Dataset Tiers

| Tier | Source | Notes |
|------|--------|-------|
| T0 | In-repo smoke (10 instances) | Always available, no download |
| T1 | BFCL v4 simple_python + multiple | Manual JSON download from Berkeley (see `docs/RUN_REAL.md`) |
| T2 | BFCL v4 parallel + parallel_multiple | Manual JSON download |
| T6 | BFCL v4 irrelevance | Manual JSON download |
| T3 | ToolACE (`Team-ACE/ToolACE`) | CC-BY-NC 4.0 — loaded live from the HF hub at eval time via `load_toolace()`, never redistributed by this repo. Multi-turn conversations reduced to the first (user, assistant) exchange; ground truth is Python-call-list syntax with function names that may contain spaces (`src/quantcall/datasets/toolace.py`) |
| T4 | xLAM ungated mirror (`minpeter/xlam-function-calling-60k-parsed`) | NC/gated — manifest-only; gated Salesforce source behind `use_gated_xlam: true` |
| T5 | Hermes function-calling v1 (`teknium/hermes-function-calling-v1`) | Apache 2.0; bundles glaive-function-calling-5k (credit both); reconstructed from source via manifest (not redistributed) |

---

## Supported Backends

| Backend key | Quant formats | Install |
|-------------|--------------|---------|
| `mock` | — | built-in |
| `llama-cpp` | GGUF Q4/Q5/Q8 | `uv sync --extra llama-cpp` |
| `transformers` | fp16, 8-bit, 4-bit (bitsandbytes) | `uv sync --extra transformers` |
| `vllm` | AWQ, GPTQ | `uv sync --extra vllm` |
| `openai` | any (remote endpoint) | `uv sync --extra openai` |

---

## Config Reference

```yaml
backend: llama-cpp          # mock | llama-cpp | transformers | vllm | openai
model: /path/to/model.gguf  # model path or HF repo ID
quant: Q4_K_M               # fp16 | Q8_0 | Q5_K_M | Q4_K_M | AWQ | GPTQ
tiers: [T1, T6]             # dataset tiers to evaluate
sample_size: 200            # instances per tier (null = all)
seed: 42
decoding: free              # free | constrained (GBNF, see docs/constrained_decoding_findings.md)
chat_variant: default       # default | qwen3_nothink (Hermes tool_call parser, suppresses <think>)
```

---

## Verification Gate

```bash
make verify
# Runs: ruff check, ruff format --check, pyright, pytest -q, smoke e2e
```

---

## Reproducing Results

Every `result.json` includes a manifest:

- Git commit SHA and dirty flag
- Config SHA-256
- Dataset sample SHA-256
- Hardware fingerprint (GPU name, driver, CUDA version)

---

## HuggingFace

| Artifact | URL |
|----------|-----|
| Eval suite (versioned samples) | [happynood/quantcall-suite](https://huggingface.co/datasets/happynood/quantcall-suite) |
| Results dataset (submit your runs) | [happynood/quantcall-results](https://huggingface.co/datasets/happynood/quantcall-results) |
| Live leaderboard | [happynood/quantcall-leaderboard](https://huggingface.co/spaces/happynood/quantcall-leaderboard) |

## Real GPU Evaluation

See [docs/RUN_REAL.md](docs/RUN_REAL.md) for exact GPU commands.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the PR-based submission flow.

## Citation

```bibtex
@software{quantcall2026,
  title  = {QuantCall: Benchmarking Tool-Calling Reliability Under Quantization},
  year   = {2026},
  url    = {https://github.com/Happynood/quant-toolcall-bench}
}
```

## License

MIT, see [LICENSE](LICENSE).

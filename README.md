# QuantCall — Does Quantization Break Tool Calling?

[![CI](https://github.com/Happynood/quant-toolcall-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Happynood/quant-toolcall-bench/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Leaderboard](https://img.shields.io/badge/🤗%20Space-Leaderboard-yellow)](https://huggingface.co/spaces/happynood/quantcall-leaderboard)

> *"You quantized your model to fit VRAM. Did you also quietly break its ability to call tools? On Llama-3.2-3B, dropping from Q8_0 to Q4_K_M measurably hurt argument correctness (AC) — find out on your own hardware, in one command."*

A reproducible benchmark measuring the degradation of function-calling / structured-output in open-weight LLMs under quantization and across inference backends.

---

## Leaderboard

First real result, run on an RTX 3050 Laptop GPU (4GB VRAM) against BFCL v4
(T1 simple/multiple + T6 irrelevance, n=200/seed, 3 seeds, greedy decoding).
fp16 does not fit in 4GB, so **Q8_0 is the Δ reference precision** here —
this is *not* a full-precision comparison. With only 3 repeats per quant
level, the bootstrap 95% CIs below are wide; treat these as indicative,
not final. Raw per-seed results: [🤗 quantcall-results dataset](https://huggingface.co/datasets/happynood/quantcall-results).

| Model | Quant | Backend | SVR | TSA | AC | Abst | FCR |
|-------|-------|---------|-----|-----|----|------|-----|
| Llama-3.2-3B-Instruct | Q8_0 (ref) | llama-cpp | 0.345 [0.330, 0.365] | 0.683 [0.655, 0.700] | 0.271 [0.252, 0.290] | 0.005 | 0.326 [0.325, 0.328] |
| Llama-3.2-3B-Instruct | Q4_K_M | llama-cpp | 0.340 [0.320, 0.355] | 0.680 [0.655, 0.700] | 0.247 [0.244, 0.250] | 0.000 | 0.317 [0.311, 0.322] |

Δ (Q8_0 → Q4_K_M), mean of 3 seeds: AC **−8.7% relative** (0.271 → 0.247), FCR −2.9% relative. Live table: [🤗 Space](https://huggingface.co/spaces/happynood/quantcall-leaderboard).

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
| T3 | ToolACE (`Team-ACE/ToolACE`) | CC-BY-NC 4.0 — manifest-only, no redistribution |
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
decoding: free              # free | gbnf | xgrammar | outlines
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

MIT

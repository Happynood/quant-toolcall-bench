# QuantCall — Does Quantization Break Tool Calling?

> *"You quantized your model to fit VRAM. Did you also quietly break its ability to call tools? Find out — on your own hardware, in one command."*

A reproducible benchmark measuring the degradation of function-calling / structured-output in open-weight LLMs under quantization and across inference backends.

---

## Leaderboard

*No results yet. Run a sweep and submit a PR to populate the leaderboard.*

| Model | Quant | Backend | SVR | TSA | AC | Abst | FCR | ΔFCR |
|-------|-------|---------|-----|-----|----|------|-----|------|
| — | — | — | — | — | — | — | — | — |

---

## Quickstart

```bash
# Install
pip install quantcall   # or: uv pip install quantcall

# Run against a local Ollama / llama.cpp server (zero-GPU path)
quantcall run --config configs/smoke.yaml

# Full quant sweep (requires a downloaded GGUF model)
quantcall sweep --model path/to/model.gguf --quants Q4_K_M,Q5_K_M,Q8_0,fp16
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| **SVR** | Schema-Validity Rate — are all emitted tool-calls structurally valid? |
| **TSA** | Tool-Selection Accuracy — correct tool names selected? |
| **AC** | Argument Correctness — correct argument values (AST-match)? |
| **Abst** | Abstention Accuracy — does the model correctly *not* call when irrelevant? |
| **FCR** | Function-Calling Reliability — weighted aggregate |
| **ΔFCR** | Degradation vs fp16 baseline |

---

## Supported Backends

| Backend | Quant formats | Status |
|---------|--------------|--------|
| mock | — | ✓ (CI/plumbing) |
| llama-cpp | GGUF Q4/Q5/Q8 | Phase 1 |
| transformers | fp16, 8-bit, 4-bit | Phase 3 |
| vllm | AWQ, GPTQ | Phase 3 |
| openai-compatible | any | Phase 3 |

---

## Submit Your Model

See [CONTRIBUTING.md](CONTRIBUTING.md) for the PR-based submission flow.

---

## Citation

```bibtex
@software{quantcall2026,
  title  = {QuantCall: Benchmarking Tool-Calling Reliability Under Quantization},
  year   = {2026},
  url    = {https://github.com/Happynood/quant-toolcall-bench}
}
```

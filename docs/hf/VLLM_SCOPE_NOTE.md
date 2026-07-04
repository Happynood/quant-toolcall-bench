# vLLM backend: implemented, not run on this hardware

`src/quantcall/backends/vllm_backend.py` implements `VLLMBackend` against
vLLM's real offline `LLM.chat(messages, sampling_params=..., tools=...)` API
(verified against current vLLM docs — see
`docs/examples/tool_calling/chat_with_tools_offline` — not guessed). It is
wired into `cli.py`'s `_build_backend` for `backend: vllm` and covered by
unit tests (`tests/test_vllm_backend.py`) using a mocked `vllm` module, the
same pattern used for the other optional-dependency backends.

**It has not been run for real on this project's hardware (an RTX 3050
Laptop GPU, 4GB VRAM).** vLLM's PagedAttention KV-cache allocator reserves a
large, mostly-fixed chunk of VRAM up front (governed by
`gpu_memory_utilization`) on top of model weights, and the project has no
evidence this fits in the ~3.7GB actually free on this card even for the
smallest models tested elsewhere in this benchmark (Qwen3-0.6B, Llama-3.2-1B).
Installing vLLM's dependency tree is also large (CUDA-version-pinned torch,
xformers/flash-attn wheels) relative to the time budget for this pass.

This is a disclosed scope reduction, not a fabricated or silently skipped
result: `quant: vllm` in a config now dispatches to a real, correct
implementation instead of crashing with `ModuleNotFoundError` (its state
before this pass), but no `results/*.json` file with `backend: vllm` exists
in this repository, and none should be assumed to. Anyone with a GPU that
has more headroom (~8GB+ is the commonly reported practical floor for vLLM)
can run it directly:

```bash
uv sync --extra vllm
uv run quantcall run --config <a config with backend: vllm> --output results/....json
```

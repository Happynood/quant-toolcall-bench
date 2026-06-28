# Running Real GPU Evaluations

This guide covers running QuantCall against actual quantized models on a GPU. All
commands are exact — copy-paste ready.

## Prerequisites

- Python 3.11+, `uv` installed (`pip install uv`)
- CUDA-capable GPU (8 GB+ VRAM for Q4_K_M of a 7B model)
- A downloaded GGUF model file (see below for download commands)

## Install with llama.cpp backend

```bash
git clone https://github.com/Happynood/quant-toolcall-bench
cd quant-toolcall-bench
uv sync --extra llama-cpp
```

## Download a GGUF model

```bash
# Example: Qwen2.5-7B-Instruct Q4_K_M from Hugging Face
pip install huggingface-hub
huggingface-cli download \
    Qwen/Qwen2.5-7B-Instruct-GGUF \
    qwen2.5-7b-instruct-q4_k_m.gguf \
    --local-dir models/
```

## Download BFCL data (required for T1/T2/T6)

BFCL is not loadable via `datasets.load_dataset`. Download files manually:

```bash
mkdir -p data/bfcl
# Replace with the actual Berkeley BFCL release URL when available
# https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
# Download: gorilla_openfunctions_v1_test_simple.json
# Download: gorilla_openfunctions_v1_test_multiple_function.json
# Download: gorilla_openfunctions_v1_test_irrelevance.json
# Place them in data/bfcl/
```

## Write a config

Create `configs/qwen2.5-7b-q4.yaml`:

```yaml
backend: llama-cpp
model: models/qwen2.5-7b-instruct-q4_k_m.gguf
quant: Q4_K_M
tiers: [T1, T6]
sample_size: 200
seed: 42
decoding: free
```

## Run the evaluation

```bash
mkdir -p results

uv run quantcall run \
    --config configs/qwen2.5-7b-q4.yaml \
    --output results/qwen2.5-7b-q4_k_m.json \
    --manifest results/qwen2.5-7b-q4_k_m.manifest.json

# Check the summary
cat results/qwen2.5-7b-q4_k_m.json | python -m json.tool | grep -E '"(svr|tsa|ac|fcr)'
```

## Run a quantization sweep

```bash
for QUANT in fp16 Q8_0 Q5_K_M Q4_K_M; do
    uv run quantcall run \
        --config configs/qwen2.5-7b-${QUANT}.yaml \
        --output results/qwen2.5-7b-${QUANT}.json \
        --manifest results/qwen2.5-7b-${QUANT}.manifest.json
done
```

## Build the leaderboard from results

```bash
uv run quantcall leaderboard --results-dir results/ --output-dir docs/leaderboard/
cat docs/leaderboard/leaderboard.md
```

## Run with constrained decoding (GBNF)

```yaml
# In your config:
decoding: gbnf
```

```bash
uv run quantcall run \
    --config configs/qwen2.5-7b-q4-gbnf.yaml \
    --output results/qwen2.5-7b-q4_k_m-gbnf.json
```

Compare CDR between free and constrained runs to see how much constrained
decoding recovers function-calling reliability.

## Submit results to the leaderboard

After verifying your results:

1. Check that the manifest is included in your result JSON.
2. Open a PR adding your result files under `results/`.
3. The CI pipeline will validate the manifest and regenerate the leaderboard.

## Troubleshooting

**`llama_cpp` import error**: Install with `uv sync --extra llama-cpp`. On some
systems you may need to install `llama-cpp-python` with CUDA support manually:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

**Out of VRAM**: Reduce `sample_size` in your config or use a smaller quant (Q4_K_S).

**Slow runs**: Set `sample_size: 50` for a quick smoke check, then run full eval overnight.

# Running Real GPU Evaluations

This guide covers running QuantCall against actual quantized models on a GPU. All
commands below were run end-to-end on an RTX 3050 Laptop GPU (4 GB VRAM) to
produce the first real leaderboard entries — they are exact, copy-paste ready.

## Prerequisites

- Python 3.11+, `uv` installed (`pip install uv`)
- A CUDA-capable NVIDIA GPU. Only the driver is required — the CUDA *toolkit*
  does not need to be installed (see "GPU offload without the CUDA toolkit" below).
- A downloaded GGUF model file (see below for download commands)

## Install with the llama.cpp backend

```bash
git clone https://github.com/Happynood/quant-toolcall-bench
cd quant-toolcall-bench
uv sync --extra llama-cpp
```

`pyproject.toml`'s `llama-cpp` extra pulls in `nvidia-cuda-runtime-cu12` and
`nvidia-cublas-cu12` pip wheels alongside `llama-cpp-python`. These provide
`libcudart.so.12` / `libcublas.so.12` for machines that only have the NVIDIA
driver installed (common on laptops), not the full CUDA toolkit.

### GPU offload without the CUDA toolkit

If `llama_cpp` fails to import with
`libcudart.so.12: cannot open shared object file`, this is expected on driver-only
systems. `LlamaCppBackend` (`src/quantcall/backends/llama_cpp.py`) works around
this automatically by preloading the CUDA `.so` files from the pip-installed
`nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12` packages with `ctypes.RTLD_GLOBAL`
before importing `llama_cpp` — no manual steps needed once the extras are installed.

## Download a GGUF model

```bash
pip install huggingface-hub
huggingface-cli download \
    bartowski/Llama-3.2-3B-Instruct-GGUF \
    --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
    --local-dir ~/models/
huggingface-cli download \
    bartowski/Llama-3.2-3B-Instruct-GGUF \
    --include "Llama-3.2-3B-Instruct-Q8_0.gguf" \
    --local-dir ~/models/
```

`fp16` GGUF (~6 GB) does not fit a 4 GB card — `Q8_0` (~3.2 GB) is the least-lossy
precision that fits, and is used as the Δ reference in the current leaderboard.

## Download BFCL v4 data (required for T1/T2/T6)

BFCL is **not** loadable via `datasets.load_dataset`. The dataset moved to v4
with new filenames and a nested question format; download the files manually:

```bash
mkdir -p data/bfcl/possible_answer

BASE="https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data"

curl -sL "$BASE/BFCL_v4_simple_python.json" -o data/bfcl/BFCL_v4_simple_python.json
curl -sL "$BASE/BFCL_v4_multiple.json" -o data/bfcl/BFCL_v4_multiple.json
curl -sL "$BASE/BFCL_v4_irrelevance.json" -o data/bfcl/BFCL_v4_irrelevance.json

curl -sL "$BASE/possible_answer/BFCL_v4_simple_python.json" \
    -o data/bfcl/possible_answer/BFCL_v4_simple_python.json
curl -sL "$BASE/possible_answer/BFCL_v4_multiple.json" \
    -o data/bfcl/possible_answer/BFCL_v4_multiple.json
```

`irrelevance` (T6, abstention) has no `possible_answer` file by design — no
call is ever expected. v4 stores each ground-truth argument as a list of
acceptable values (e.g. `{"unit": ["units", ""]}`); the adapter
(`src/quantcall/datasets/bfcl.py::_first_value`) takes the first non-empty
value as canonical, which is a conservative choice — AC may slightly
undercount correct calls that use an alternative value, but the
cross-quantization degradation *trend* remains valid.

## Write a config

```yaml
# configs/llama32-3b-q4.yaml
backend: llama-cpp
model: /home/you/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
quant: Q4_K_M
decoding: free
tiers: [T1, T6]
sample_size: 200
seed: 42
temperature: 0.0
bfcl_data_dir: data/bfcl

llama_cpp:
  n_ctx: 1024
  n_gpu_layers: -1
  max_tokens: 256
  verbose: false
```

**VRAM note:** on a 4 GB card, `n_ctx: 2048` was enough to make `Q8_0` fail
with a CUDA OOM during compute-buffer allocation (`ggml_gallocr_reserve_n_impl:
failed to allocate CUDA0 buffer`). Dropping to `n_ctx: 1024` / `max_tokens: 256`
fixed it for both `Q8_0` and `Q4_K_M`. If you have more VRAM, raise these.

## Run the evaluation

```bash
mkdir -p results

uv run quantcall run \
    --config configs/llama32-3b-q4.yaml \
    --output results/llama32-3b-q4_k_m.json \
    --manifest results/llama32-3b-q4_k_m.manifest.json

# Check the summary
cat results/llama32-3b-q4_k_m.json | python -m json.tool | grep -E '"(svr|tsa|ac|fcr)'
```

## Run a quantization sweep with repeats

Use multiple seeds to get more than one sample per quant level — a single
run gives you a point estimate with no variance information.

```bash
for QUANT in Q8_0 Q4_K_M; do
    for SEED in 42 43 44; do
        uv run quantcall run \
            --config configs/sweep-${QUANT}-s${SEED}.yaml \
            --output results/sweep/${QUANT}_s${SEED}.json \
            --manifest results/sweep/${QUANT}_s${SEED}.manifest.json
    done
done
```

Then compute bootstrap 95% CIs and the Δ vs a reference precision:

```bash
uv run python3 scripts/aggregate_sweep.py
```

This reads `results/sweep/*.json`, resamples the per-seed run means, and
writes `results/leaderboard/quant_delta_report.json`. **With only 3 repeats
per quant level the resulting CIs are wide** — treat them as indicative, not
a precise estimate. Increase `SEEDS` and rerun for a tighter CI.

## Build the leaderboard from results

```bash
uv run quantcall leaderboard results/sweep --output-dir results/leaderboard
cat results/leaderboard/leaderboard.md
```

(`leaderboard` takes the results directory as a positional argument, not
`--results-dir`.)

## Publish results to the HF dataset

Results are not committed to git — they're published to the
`happynood/quantcall-results` HF dataset that the Space reads live. See
`docs/PUBLISH_HF.md` §3 for the exact `hf upload` command. Before publishing,
scrub any local filesystem paths out of the `Model` column (the `config.model`
field defaults to the local GGUF path).

## Run with constrained decoding (GBNF)

```yaml
# In your config:
decoding: constrained
```

Compare FCR between free and constrained runs to see how much constrained
decoding recovers function-calling reliability under quantization.

## Troubleshooting

**`llama_cpp` import error / `libcudart.so.12: cannot open shared object file`**:
Expected on driver-only systems; see "GPU offload without the CUDA toolkit" above.
Confirm the extra is installed: `uv sync --extra llama-cpp`.

**`ValueError: Failed to create llama_context`**: Usually a CUDA OOM during
compute-buffer allocation on small VRAM budgets. Lower `n_ctx` and/or
`max_tokens` in the config until the model loads (see the VRAM note above).

**Slow runs**: Set `sample_size: 50` for a quick smoke check before running
the full sweep.

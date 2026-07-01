# Running Real GPU Evaluations

This guide covers running QuantCall against actual quantized models on a GPU. All
commands below were run end-to-end on an RTX 3050 Laptop GPU (4 GB VRAM) to
produce the real leaderboard entries — they are exact, copy-paste ready.

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

The primary model is Qwen3-0.6B, chosen specifically because its fp16 (bf16)
weights (~1.5 GB) fit a 4 GB card with room to spare, so a genuine fp16
baseline is possible (unlike a 3B-class model, where fp16 doesn't fit). The
official `Qwen/Qwen3-0.6B-GGUF` repo only ships `Q8_0`; bartowski's repo has
the full quant ladder including bf16 (used here as the `fp16` value in
configs):

```bash
pip install huggingface-hub
for QUANT_FILE in Q4_K_M Q5_K_M Q8_0 bf16; do
    huggingface-cli download \
        bartowski/Qwen_Qwen3-0.6B-GGUF \
        --include "Qwen_Qwen3-0.6B-${QUANT_FILE}.gguf" \
        --local-dir ~/models/
done
# Secondary model, same quant ladder minus bf16 (see below):
for QUANT_FILE in Q4_K_M Q5_K_M Q8_0; do
    huggingface-cli download \
        bartowski/Qwen_Qwen3-1.7B-GGUF \
        --include "Qwen_Qwen3-1.7B-${QUANT_FILE}.gguf" \
        --local-dir ~/models/
done
```

### Qwen3-1.7B fp16 does not fit a 4 GB card — verified, not assumed

`bf16` (`Qwen_Qwen3-1.7B-bf16.gguf`, ~4.07 GB) was actually loaded against
the RTX 3050 Laptop GPU (4096 MiB total). It fails with a real CUDA OOM
during KV-cache allocation at `n_ctx=4096` **and** at `n_ctx=2048`; it only
succeeds at `n_ctx=512`, too small to be a fair/comparable eval context
for BFCL's tool-schema-heavy prompts:

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 448.00 MiB on device 0: cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate CUDA0 buffer of size 469762048
llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache
```

Because a usable-context fp16 run isn't possible for the 1.7B model,
**Q8_0 is its Δ reference precision** — labeled explicitly as
`baseline_quant` in `leaderboard.csv`, never silently assumed to be fp16.
Qwen3-0.6B's fp16 *does* fit at `n_ctx=4096` and is used as its baseline.
The baseline is picked automatically per model by the highest-ranked quant
actually present (`src/quantcall/report/published.py::PRECISION_RANK`).

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

## Qwen3-specific config: `chat_variant: qwen3_nothink`

Qwen3 emits a `<think>...</think>` reasoning block by default. Set
`chat_variant: qwen3_nothink` to append `/no_think` to the user turn
(suppresses the reasoning content — verified empirically to produce an
*empty* `<think></think>` block, not assumed from docs, see
`docs/parser_audit.md`) and to parse tool calls with the Hermes-style
`<tool_call>{...}</tool_call>` parser (`HermesXmlParser`) instead of the
generic `raw_json` parser. The model's own GGUF-embedded chat template is
used automatically (no `--jinja` flag needed — that's llama-cpp-python's
default when `chat_format` is left unset).

```yaml
# configs/qwen3-sweep/qwen3-0.6b-Q4_K_M-s0.yaml
backend: llama-cpp
model: /home/you/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf
quant: Q4_K_M
decoding: free
chat_variant: qwen3_nothink
tiers: [T1, T6]
sample_size: 200
seed: 0
temperature: 0.0
bfcl_data_dir: data/bfcl

llama_cpp:
  n_ctx: 4096
  n_gpu_layers: -1
  max_tokens: 256
  verbose: false
```

## Run the evaluation

```bash
mkdir -p results

uv run quantcall run \
    --config configs/qwen3-sweep/qwen3-0.6b-Q4_K_M-s0.yaml \
    --output results/qwen3-0.6b-Q4_K_M-s0.json \
    --manifest results/qwen3-0.6b-Q4_K_M-s0.manifest.json

# Check the summary
cat results/qwen3-0.6b-Q4_K_M-s0.json | python -m json.tool | grep -E '"(svr|tsa|ac|fcr)'
```

## Run a quantization sweep with repeats

Use multiple seeds (0, 1, 2) to get more than one sample per quant level —
a single run gives you a point estimate with no variance information.
`configs/qwen3-sweep/` has 21 ready-made configs (0.6B × [fp16, Q8_0,
Q5_K_M, Q4_K_M] × 3 seeds; 1.7B × [Q8_0, Q5_K_M, Q4_K_M] × 3 seeds):

```bash
for CFG in configs/qwen3-sweep/*.yaml; do
    NAME=$(basename "$CFG" .yaml)
    uv run quantcall run \
        --config "$CFG" \
        --output "results/qwen3-sweep/${NAME}.json" \
        --manifest "results/qwen3-sweep/${NAME}.manifest.json"
done
```

## Build the leaderboard from results

`quantcall leaderboard` computes bootstrap 95% CIs and the Δ vs the
best-available baseline quant itself (see `src/quantcall/report/published.py`)
and writes both published files:

```bash
uv run quantcall leaderboard results/qwen3-sweep --output-dir results/leaderboard
cat results/leaderboard/leaderboard.md
```

(`leaderboard` takes the results directory as a positional argument, not
`--results-dir`.) **With only 3 repeats per quant level the resulting CIs are
wide** — treat them as indicative, not a precise estimate. Increase the
number of seeds and rerun for a tighter CI.

For per-metric (ΔSVR/ΔAC/ΔFCR) bootstrap CIs and significance verdicts
(does the CI exclude zero?), rather than just the ΔFCR the leaderboard.csv
carries:

```bash
uv run python3 scripts/delta_significance.py
```

## Publish results to the HF dataset

Results are not committed to git — they're published to the
`happynood/quantcall-results` HF dataset that the Space reads live. See
`docs/PUBLISH_HF.md` §3 for the exact `hf upload` commands.

`quantcall leaderboard` sanitizes `config.model` automatically
(`src/quantcall/report/published.py::sanitize_model_name`) — local GGUF paths
like `/home/x/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf` are stripped to a canonical
name (`Qwen3-0.6B`) so every quant of one model groups into the same
`(model, backend, decoding, tier)` scope for baseline/delta computation. No
manual pre-publish sanitization step is needed.

**"Done" for any HF artifact requires re-fetching it from the hub after
pushing and checking the real contents** — a local file being written is not
proof anything actually reached the hub.

## Run with constrained decoding (GBNF)

```yaml
# In your config:
decoding: constrained
```

`configs/qwen3-constrained/` has 7 ready-made configs (1 seed each). This
builds a GBNF grammar per instance (`src/quantcall/decoding/gbnf.py::
build_tool_call_grammar`) that forces the model's output to match one of the
available tools' exact JSON Schema. **Real finding, not the naive
expectation:** this did not improve SVR or AC for Qwen3 in our testing, and
Abstention/FCR crater under it because the grammar has no "abstain"
alternative (it always forces a `<tool_call>`). See
`docs/constrained_decoding_findings.md` for the full analysis, including a
GBNF-parser segfault we found and fixed along the way
(`src/quantcall/decoding/gbnf.py`'s rule-naming had to avoid mixing `_`
and `-`, which crashes llama.cpp's grammar parser outright).

```bash
for CFG in configs/qwen3-constrained/*.yaml; do
    NAME=$(basename "$CFG" .yaml)
    uv run quantcall run \
        --config "$CFG" \
        --output "results/qwen3-constrained/${NAME}.json" \
        --manifest "results/qwen3-constrained/${NAME}.manifest.json"
done
uv run python3 scripts/cdr_analysis.py
```

## Troubleshooting

**`llama_cpp` import error / `libcudart.so.12: cannot open shared object file`**:
Expected on driver-only systems; see "GPU offload without the CUDA toolkit" above.
Confirm the extra is installed: `uv sync --extra llama-cpp`.

**`ValueError: Failed to create llama_context`**: Usually a CUDA OOM during
compute-buffer or KV-cache allocation on small VRAM budgets. Lower `n_ctx`
and/or `max_tokens` in the config until the model loads.

**Grammar parse crash (exit 139) under `decoding: constrained`**: Already
fixed in this repo (see `docs/constrained_decoding_findings.md`), but if you
hit a similar segfault with a custom grammar, suspect rule names that mix
`_` and `-`.

**Slow runs**: Set `sample_size: 50` for a quick smoke check before running
the full sweep.

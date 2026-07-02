# GPU environment regression (self-inflicted, found and fixed mid-session)

While investigating whether the `datasets` extra was installed (for the
ToolACE/T3 adapter), a plain `uv sync` (no `--extra` flags) was run. This
resolved dependencies against the base `[project.dependencies]` list only,
which **uninstalled `llama-cpp-python`, `nvidia-cuda-runtime-cu12`, and
`nvidia-cublas-cu12`**. Running `uv sync --extra llama-cpp --extra datasets`
immediately after reinstalled `llama-cpp-python`, but from PyPI's default
wheel for this platform -- which is **CPU-only** (no `libggml-cuda.so` in
`llama_cpp_python.libs/`). Verifying with `python3 -c "import llama_cpp"`
is not sufficient to catch this: the import succeeds either way. The only
way to actually confirm CUDA support is present is to load a real model
with `n_gpu_layers=-1, verbose=True` and check for lines like
`load_tensors: layer N assigned to device CUDA0` (a CPU-only build reports
`device CPU` for every layer, and `llama_context: backend_ptrs.size() = 1`).

This was caught because a Llama-3.2-1B fp16 free-decoding run that should
take ~3 minutes was still running after 15+ minutes at 850%+ CPU with 0%
GPU utilization (`nvidia-smi --query-gpu=utilization.gpu`).

**Fix applied:** `uv`'s local package cache
(`~/.cache/uv/archive-v0/<hash>/llama_cpp_python.libs/`) still had an
earlier CUDA-enabled build of the exact same `llama-cpp-python==0.3.31`
from before the mishap (multiple cached builds coexist under different
content hashes). Copied that cached build's `llama_cpp/`,
`llama_cpp_python-0.3.31.dist-info/`, and `llama_cpp_python.libs/`
directories directly into `.venv/lib/python3.12/site-packages/`, replacing
the broken CPU-only install. Verified via the same
`load_tensors: ... device CUDA0` check before trusting any further runs.

**Consequence:** the constrained-decoding sweep (`configs/qwen3-constrained/`)
had already been run once against the broken CPU-only build partway through
(some configs on CPU, and later configs contending for GPU with a
simultaneously-running Llama-3.2-1B sweep that also silently degraded to
CPU). Rather than try to determine which specific result files were
affected, **the entire 7-config constrained sweep was deleted and re-run
from scratch** once GPU support was confirmed restored. The numbers in
`docs/constrained_decoding_findings.md` are from that clean re-run.

**Takeaway for future sessions:** never run a bare `uv sync` in this repo
without the extras this project needs (`--extra llama-cpp --extra
datasets`, or check `pyproject.toml`'s `[project.optional-dependencies]`
for the full current list) -- it silently drops GPU support. If in doubt
after any `uv sync`, verify CUDA is really active with the
`verbose=True` / `device CUDA0` check above before trusting any GPU-timed
result, not just that `import llama_cpp` succeeds.

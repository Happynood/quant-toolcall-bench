# Backend verification: transformers, openai, vllm

Before this pass, `cli.py` already dispatched `backend: transformers` /
`backend: openai` / `backend: vllm` to `quantcall.backends.hf` /
`quantcall.backends.openai_endpoint` / `quantcall.backends.vllm_backend` —
but none of those three modules existed. Any config using them crashed with
`ModuleNotFoundError` before a single instance ran. This pass implemented
all three (see `src/quantcall/backends/{hf,openai_endpoint,vllm_backend}.py`)
and, for the two that are feasible on this project's hardware, ran real
sweeps and sanity-checked the numbers before trusting them — the same
discipline used earlier in this project for the Llama-3.2-1B parser check.

## transformers backend — Qwen3-0.6B, bf16

First real run against `Qwen/Qwen3-0.6B` crashed immediately:

```
OSError: Unknown scheme for proxy URL URL('socks://127.0.0.1:12334')
```

`httpx` (used internally by `huggingface_hub`) rejects a bare `socks://`
`ALL_PROXY` value that this machine's shell exports. Fixed in
`HFBackend.__init__` by clearing `ALL_PROXY`/`all_proxy` before importing
`transformers`/`torch` (same fix the reference `llm-inference-benchmark`
repo already applies in its own `hf.py`). Regression test:
`tests/test_hf_backend.py::test_hf_backend_clears_socks_proxy_env_vars`.

After the fix, a real 3-seed sweep (T1+T6, n=200/seed, `configs/hf-sweep/`)
gave SVR 0.865 / 0.880 / 0.885 (mean 0.877), AC 0.626 / 0.568 / 0.618 (mean
0.604) — matching the `llama-cpp` fp16 numbers for the same model
(SVR 0.877, AC 0.605) closely enough to be a real, not-cherry-picked
confirmation that backend choice alone doesn't move these metrics when
precision is held constant. See the README's "Does the backend itself
matter" section for the full comparison table.

## openai backend — Gemma via a local LM Studio server

First real call against a locally running LM Studio server
(`http://127.0.0.1:1234/v1`, model `gemma-4-e4b`) returned SVR=0.0 on a
5-instance sanity check — implausible, so it was not trusted. A direct
`curl` to the same endpoint showed the model *did* return correctly-formed
native `tool_calls`, so the bug wasn't in tool-calling itself. Root cause:

```python
>>> import urllib.request
>>> urllib.request.proxy_bypass_environment("127.0.0.1")
False
```

Python's stdlib `no_proxy` parsing does not understand CIDR notation
(`127.0.0.0/8`, as exported by this shell's `no_proxy` env var) — so every
request to `127.0.0.1:1234` was silently being routed through the system's
configured `HTTP_PROXY` (`127.0.0.1:12334`, an unrelated local proxy tool),
which returned an HTTP 502 instead of ever reaching LM Studio. Fixed in
`OpenAIEndpointBackend` by bypassing any configured proxy specifically for
loopback hosts (`localhost`, `127.0.0.1`, `::1`, `127.x.x.x`) while still
honoring a system proxy for genuinely remote endpoints. Regression tests:
`tests/test_openai_endpoint_backend.py::test_open_url_uses_proxy_free_opener_for_loopback_host`
and the paired `..._uses_regular_urlopen_for_remote_host` test.

After the fix, the same 5-instance check produced SVR=0.8 (plausible), and
a real 3-seed sweep (T1+T6, n=200/seed, `configs/openai-sweep/`) gave
SVR 0.865 / 0.885 / 0.885, AC 0.604 / 0.634 / 0.679 — both published to the
leaderboard.

## vllm backend — implemented, not run

See [docs/hf/VLLM_SCOPE_NOTE.md](VLLM_SCOPE_NOTE.md). Implemented against
vLLM's real offline `LLM.chat(messages, sampling_params=..., tools=...)`
API (checked against current vLLM docs, not guessed), unit-tested with a
mocked `vllm` module, but not run for real — this project's 4GB laptop GPU
has no evidence of fitting vLLM's PagedAttention KV-cache reservation
alongside model weights, and no `results/*.json` file with `backend: vllm`
exists in this repository.

## Why this matters for the "no fabricated numbers" rule

Both real bugs above would have produced a plausible-looking but *wrong*
published number (SVR=0.0 for Gemma, or a crash that might have tempted a
smaller/different sanity config) had the sanity-check step been skipped.
Neither was a tool-calling or parsing defect — both were local network/proxy
configuration issues specific to this machine — which is exactly why the
project's standing rule is to sanity-check a small real run before trusting
a full sweep, rather than trusting a number just because the code ran
without an exception.

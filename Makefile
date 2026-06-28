.PHONY: install install-llama-cpp install-transformers install-vllm \
        test lint format typecheck verify \
        run clean

# ── Dependencies ──────────────────────────────────────────────────────────────

install:
	uv sync

install-llama-cpp:
	uv sync --extra llama-cpp

install-llama-cpp-cuda:
	CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra llama-cpp

install-transformers:
	uv sync --extra transformers

install-vllm:
	uv sync --extra vllm

install-all:
	uv sync --all-extras

# ── Quality gate (must be green before advancing any phase) ──────────────────

verify: lint-check format-check typecheck test smoke

lint-check:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run pyright

test:
	uv run pytest -q

smoke:
	uv run quantcall run --config configs/smoke.yaml --output /tmp/qc-smoke-result.json --manifest /tmp/qc-smoke-manifest.json
	@python -c "import json, sys; d=json.load(open('/tmp/qc-smoke-result.json')); sys.exit(0 if 'svr' in d and 'manifest' in d for _ in [1] else 1)" 2>/dev/null || \
	 python -c "import json; d=json.load(open('/tmp/qc-smoke-result.json')); assert 'svr' in d, 'result.json missing svr'; print('smoke OK: svr=' + str(d['svr']))"

# ── Dev helpers ───────────────────────────────────────────────────────────────

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

# ── Benchmark runs ───────────────────────────────────────────────────────────

run:
	uv run quantcall run --config configs/smoke.yaml

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -f /tmp/qc-smoke-*.json

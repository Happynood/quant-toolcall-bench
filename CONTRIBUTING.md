# Contributing

## Submitting Benchmark Results

To add a model to the leaderboard:

1. Run the evaluation on your own hardware:
   ```bash
   quantcall run \
     --config configs/your_config.yaml \
     --output results/your_model_quant.json \
     --manifest results/your_model_quant.manifest.json
   ```

2. Verify the result file includes a manifest (git SHA, config hash, hardware fingerprint).

3. Open a PR adding only `results/your_model_quant.json` and `results/your_model_quant.manifest.json`. Do not edit the leaderboard table manually — it is generated from result files.

## Code Contributions

### Setup

```bash
git clone https://github.com/Happynood/quant-toolcall-bench
cd quant-toolcall-bench
pip install uv
uv sync --dev
```

### Workflow

1. Create a feature branch.
2. Write failing tests first (TDD).
3. Implement until tests pass.
4. Run `make verify` — must be green before any PR.
5. Open a PR with a clear description.

### Verification Gate

```bash
make verify
```

This runs: `ruff check`, `ruff format --check`, `pyright`, `pytest -q`, and the smoke end-to-end test.

## Hard Rules

- Never fabricate, hardcode, or guess metric values. Real numbers come only from real model runs.
- The leaderboard ships empty and is populated only from verified `result.json` files.
- BFCL files must not be loaded via `datasets.load_dataset` — always use the manual JSONL reader.
- Every result file must include a complete manifest for reproducibility.

## Dataset Adapters

To add a new dataset tier (e.g., T7):

1. Add a `src/quantcall/datasets/your_dataset.py` with a `normalize_*_instance()` function returning `NormalizedInstance`.
2. Register the tier in `src/quantcall/config.py` and `src/quantcall/cli.py`.
3. Add tests in `tests/test_your_dataset.py`.
4. Document the tier in `README.md`.

## Adding a Backend

1. Create `src/quantcall/backends/your_backend.py` inheriting from `Backend`.
2. Implement `generate_toolcall(messages, tools) -> ToolCallResult`.
3. Register in `src/quantcall/cli.py` (`_build_backend`).
4. Add an optional dependency group in `pyproject.toml`.

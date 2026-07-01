# Published Results Schema

This file is the single source of truth for the two CSV files QuantCall
publishes to the `happynood/quantcall-results` HF dataset. The column lists
below must exactly match `RUNS_COLS` / `LEADERBOARD_COLS` in
`src/quantcall/report/published.py` — `tests/test_report.py::test_no_schema_drift`
parses this file and fails the build if they diverge.

Both files are produced by `quantcall leaderboard <results_dir> --output-dir <dir>`.

## `runs.csv` — one row per real run (per seed)

| Column | Type | Description |
|--------|------|-------------|
| `model` | string | Model identifier (HF repo ID or local path) |
| `quant` | string | Quantization level: fp16, Q8_0, Q5_K_M, Q4_K_M, AWQ, GPTQ |
| `backend` | string | Inference backend: llama-cpp, transformers, vllm, openai |
| `decoding` | string | Decoding mode: free or constrained |
| `tier` | string | Dataset tier(s) evaluated, `+`-joined (e.g. `T1+T6`) |
| `seed` | int | Random seed for this run |
| `sample_size` | int | Number of instances evaluated per tier |
| `svr` | float | Schema-Validity Rate [0, 1] |
| `tsa` | float | Tool-Selection Accuracy [0, 1] |
| `ac` | float | Argument Correctness [0, 1] |
| `abstention` | float | Abstention Accuracy [0, 1] |
| `fcr` | float | Function-Calling Reliability — 0.25 × (SVR + TSA + AC + Abst) |
| `vram_gb` | float | Peak VRAM usage in GB for this run (empty if not measured) |
| `git_commit` | string | QuantCall repo commit SHA used for this run |
| `config_sha256` | string | SHA-256 of the run config |
| `dataset_sha256` | string | SHA-256 of the evaluation sample |
| `timestamp` | string | ISO-8601 UTC timestamp of the run |

## `leaderboard.csv` — one row per (model, quant, backend, decoding, tier), aggregated over seeds

| Column | Type | Description |
|--------|------|-------------|
| `model` | string | Model identifier |
| `quant` | string | Quantization level |
| `backend` | string | Inference backend |
| `decoding` | string | Decoding mode |
| `tier` | string | Dataset tier(s), `+`-joined |
| `n_seeds` | int | Number of seeds aggregated into this row |
| `fcr_mean` | float | Mean FCR across seeds |
| `fcr_ci_low` | float | Bootstrap 95% CI lower bound for FCR |
| `fcr_ci_high` | float | Bootstrap 95% CI upper bound for FCR |
| `svr_mean` | float | Mean SVR across seeds |
| `tsa_mean` | float | Mean TSA across seeds |
| `ac_mean` | float | Mean AC across seeds |
| `abstention_mean` | float | Mean Abstention across seeds |
| `vram_gb` | float | Mean peak VRAM in GB (empty if not measured by any run in the group) |
| `eta` | float | Efficiency: fcr_mean / vram_gb (empty if vram_gb is empty) |
| `delta_fcr_rel` | float | Relative FCR delta vs `baseline_quant` in the same (model,backend,decoding,tier) scope; empty for the baseline row itself |
| `delta_ac_rel` | float | Relative AC delta vs `baseline_quant` |
| `baseline_quant` | string | The Δ reference quant for this scope: fp16 if it fits and was run, otherwise the best-available quant that was run (labeled here, never silently substituted) |

## Baseline selection

The baseline quant per (model, backend, decoding, tier) scope is the
highest-ranked precision actually present in that scope, using the ranking
`fp16 > Q8_0 > Q5_K_M > Q4_K_M > (AWQ, GPTQ)`. If fp16 was not run (e.g. it
does not fit the available VRAM), the next best quant becomes the baseline —
this is always visible in the `baseline_quant` column, never hidden.

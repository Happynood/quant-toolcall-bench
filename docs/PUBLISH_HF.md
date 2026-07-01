# Publishing to HuggingFace

This document describes how to create or re-create the HuggingFace artifacts for QuantCall.

## Prerequisites

- `HF_TOKEN` environment variable set with a write-scope token for the `happynood` account
- The `hf` CLI installed (`pip install huggingface_hub[cli]`)
- SOCKS proxy issue: if `ALL_PROXY=socks://...` is set in your environment, unset it before
  running `hf` commands (httpx does not support SOCKS):
  ```bash
  unset ALL_PROXY all_proxy
  ```

## Artifacts

| Artifact | URL | Type |
|----------|-----|------|
| Eval suite | https://huggingface.co/datasets/happynood/quantcall-suite | dataset |
| Results | https://huggingface.co/datasets/happynood/quantcall-results | dataset |
| Leaderboard | https://huggingface.co/spaces/happynood/quantcall-leaderboard | space |

## 1. Create repos (first time only)

```bash
unset ALL_PROXY all_proxy
hf repos create happynood/quantcall-suite --type dataset
hf repos create happynood/quantcall-results --type dataset
hf repos create happynood/quantcall-leaderboard --type space --space-sdk gradio
```

## 2. Build and upload the suite manifest

The suite manifest records provenance (source repo, revision, sha256) without
re-hosting licensed data.

```bash
# Build manifest for T0 (always available; no external data needed)
uv run quantcall suite build \
    --tiers T0 \
    --sample-size 100 \
    --seed 42 \
    --output suite_manifest.json

# For BFCL tiers (T1/T2/T6), supply local data dir after manual download:
# uv run quantcall suite build --tiers T0,T1,T2,T6 --bfcl-data-dir data/bfcl/ ...

# Verify by materializing
uv run quantcall suite materialize --manifest suite_manifest.json

# Upload to HF
unset ALL_PROXY all_proxy
hf upload happynood/quantcall-suite suite_manifest.json suite_manifest.json --repo-type dataset
hf upload happynood/quantcall-suite data/smoke/t0_smoke.jsonl data/smoke_v1.jsonl --repo-type dataset
hf upload happynood/quantcall-suite <schemas.json> data/schemas/tool_schemas.json --repo-type dataset
```

**License decisions per tier** (coded in `src/quantcall/suite/build.py`):

| Tier | Source | License | HF artifact |
|------|--------|---------|-------------|
| T0 | In-repo smoke | MIT | smoke_v1.jsonl + schemas ✓ |
| T1/T2/T6 | BFCL (gorilla) | Apache 2.0 | manifest entries only (manual download required) |
| T3 | ToolACE | CC-BY-NC 4.0 | manifest-only, no redistribution |
| T4 | xLAM (Salesforce) | NC + gated | manifest-only, no redistribution |
| T5 | Hermes (NousResearch/teknium) | Apache 2.0 (bundles glaive-fc-5k; credit both) | manifest-only by design (adapter not yet implemented) |

## 3. Upload results dataset

The published schema (`data/runs.csv` + `data/leaderboard.csv`) is documented in
`docs/RESULTS_SCHEMA.md` and mirrored in `docs/hf/results_dataset_card.md` (the
card uploaded as the dataset's `README.md`). A repo test
(`test_hf_dataset_card_schema_matches_code`) fails the build if the card's
schema tables drift from `src/quantcall/report/published.py`.

Before publishing, scrub any local filesystem paths out of the `model` column
(the `config.model` field defaults to the local GGUF path) and build both
files with `quantcall leaderboard results/ --output-dir leaderboard/`.

```bash
unset ALL_PROXY all_proxy
hf upload happynood/quantcall-results docs/hf/results_dataset_card.md README.md --repo-type dataset
hf upload happynood/quantcall-results leaderboard/runs.csv data/runs.csv --repo-type dataset
hf upload happynood/quantcall-results leaderboard/leaderboard.csv data/leaderboard.csv --repo-type dataset
```

## 4. Upload / update the Gradio Space

```bash
unset ALL_PROXY all_proxy
hf upload happynood/quantcall-leaderboard README.md README.md --repo-type space
hf upload happynood/quantcall-leaderboard requirements.txt requirements.txt --repo-type space
hf upload happynood/quantcall-leaderboard app.py app.py --repo-type space
```

**Critical: Python version pin**

The Space README YAML must contain `python_version: "3.12"`.

`audioop` was removed from the Python standard library in Python 3.13 (PEP 594).
Some Gradio audio sub-dependencies (pydub) still import `audioop`, causing a
`ModuleNotFoundError` on startup when the Space uses Python 3.13+.

- Primary fix: `python_version: "3.12"` in Space README YAML
- Safety net: `audioop-lts; python_version >= "3.13"` in requirements.txt
- **Do not bump to 3.13 until pydub/audioop have been updated** upstream.

The GitHub Actions CI matrix also includes Python 3.13 to catch stdlib-removal
issues before they reach the Space at runtime.

## 5. After updating the Space

Check the Space Logs at:
https://huggingface.co/spaces/happynood/quantcall-leaderboard/logs

Look for a clean startup:
```
Running on public URL: https://...
```

No `ModuleNotFoundError: audioop` and no `ValueError: Unknown scheme for proxy URL`.

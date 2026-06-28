# Security Policy

## Scope

QuantCall is a local benchmarking tool. It does not expose network services or handle user authentication. Security concerns are limited to:

- Local file handling (result files, JSONL datasets)
- Optional remote model endpoints (OpenAI-compatible backend)

## Reporting a Vulnerability

To report a security issue, open a GitHub issue with the label `security`. For sensitive reports, use the GitHub private security advisory feature.

## Dependencies

Keep dependencies up to date. Run `uv sync` to install pinned versions. Review `uv.lock` before deploying in shared environments.

## Model Endpoints

When using the `openai` backend with a remote endpoint, keep API keys in environment variables — never in config YAML files committed to version control.

```bash
export QUANTCALL_API_KEY=sk-...
quantcall run --config configs/openai.yaml
```

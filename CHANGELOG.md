# Changelog

## [Unreleased]

### Added
- Phase 0: Project skeleton — Pydantic v2 config, Mock backend, T0 smoke dataset, metrics engine, Click CLI, Makefile verify gate
- Phase 1: BFCL adapter (manual JSONL reader), OpenAI tools parser, GGUF template parser, AST matcher with type coercion, delta metrics (Δ/Δ_rel), η efficiency metric, leaderboard builder (JSON/CSV/Markdown)
- Phase 2: GBNF grammar generator from JSON Schema, CDR metric (constrained-decoding recovery), T6 abstention e2e tests
- Phase 3: T3 ToolACE adapter, T4 xLAM adapter, Hermes XML parser, xLAM parser, bootstrap confidence intervals, Pareto front computation
- Phase 4: Dockerfile, GitHub Actions CI (Python 3.11 + 3.12), complete README, CONTRIBUTING, SECURITY, CHANGELOG, CITATION.cff
- Phase 5: HuggingFace publish — happynood/quantcall-suite (eval sample v1), happynood/quantcall-results (empty + schema), happynood/quantcall-leaderboard (Gradio space with filterable table + Pareto chart)
- fix: T4 default dataset switched to ungated mirror (minpeter/xlam-function-calling-60k-parsed); Salesforce gated source behind use_gated_xlam flag
- fix: FCR formula corrected to 0.25 × (SVR + TSA + AC + Abst)
- fix: README badges (CI, Python, MIT, leaderboard), Makefile python → uv run python

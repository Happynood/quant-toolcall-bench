from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from quantcall import __version__
from quantcall.config import QuantCallConfig, load_config


@click.group()
@click.version_option(version=__version__, prog_name="quantcall")
def main() -> None:
    """QuantCall — benchmark function-calling degradation under quantization."""


@main.command("run")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", default=None, type=click.Path())
@click.option("--manifest", "manifest_path", default=None, type=click.Path())
@click.option("--tier", "extra_tiers", multiple=True, help="Override tiers (repeatable)")
def run_cmd(
    config_path: str,
    output_path: str | None,
    manifest_path: str | None,
    extra_tiers: tuple[str, ...],
) -> None:
    """Run the tool-calling benchmark against a backend."""
    from quantcall.manifest import write_manifest
    from quantcall.runner import run_eval, write_result

    cfg = load_config(config_path)
    if extra_tiers:
        cfg = cfg.model_copy(update={"tiers": list(extra_tiers)})

    backend = _build_backend(cfg)
    instances = _load_instances(cfg)

    if not instances:
        click.echo("No instances loaded — check your tiers config.", err=True)
        sys.exit(1)

    click.echo(
        f"Running {len(instances)} instances | backend={cfg.backend} model={cfg.model} "
        f"quant={cfg.quant} decoding={cfg.decoding}"
    )
    result = run_eval(cfg, instances, backend, config_path=config_path)

    out = output_path or "result.json"
    write_result(result, out)
    click.echo(f"Result written to {out}")
    click.echo(
        f"  SVR={result.metrics.svr:.3f}  TSA={result.metrics.tsa:.3f}  "
        f"AC={result.metrics.ac:.3f}  Abst={result.metrics.abstention:.3f}  "
        f"FCR={result.metrics.fcr:.3f}"
    )

    if manifest_path:
        write_manifest(result.manifest, manifest_path)
        click.echo(f"Manifest written to {manifest_path}")


@main.command("sweep")
@click.option("--model", required=True)
@click.option("--quants", required=True, help="Comma-separated quant levels")
@click.option("--decoding", default="free,constrained", help="Comma-separated decoding modes")
@click.option("--base-config", default=None, type=click.Path(exists=True))
@click.option("--output-dir", default="results", type=click.Path())
def sweep_cmd(
    model: str,
    quants: str,
    decoding: str,
    base_config: str | None,
    output_dir: str,
) -> None:
    """Sweep model × quant × decoding combinations."""
    click.echo(f"[sweep stub] model={model} quants={quants} decoding={decoding}")
    click.echo("Full sweep implementation in Phase 1.")


@main.command("compare")
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json", "csv"]))
@click.option("--output", "output_path", default=None, type=click.Path())
def compare_cmd(
    result_files: tuple[str, ...],
    fmt: str,
    output_path: str | None,
) -> None:
    """Compare multiple result.json files and show delta table."""
    results: list[dict[str, Any]] = []
    for p in result_files:
        with open(p) as f:
            results.append(json.load(f))

    if fmt == "json":
        text = json.dumps(results, indent=2)
    else:
        lines = [f"{'Config':<40}  {'SVR':>6}  {'TSA':>6}  {'AC':>6}  {'FCR':>6}"]
        lines.append("-" * 70)
        for r in results:
            cfg = r.get("config", {})
            label = f"{cfg.get('model', '?')} {cfg.get('quant', '?')} {cfg.get('backend', '?')}"
            lines.append(
                f"{label:<40}  {r.get('svr', 0):.3f}  {r.get('tsa', 0):.3f}"
                f"  {r.get('ac', 0):.3f}  {r.get('fcr', 0):.3f}"
            )
        text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(text + "\n")
        click.echo(f"Written to {output_path}")
    else:
        click.echo(text)


@main.command("pareto")
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", default=None, type=click.Path())
def pareto_cmd(
    result_files: tuple[str, ...],
    output_path: str | None,
) -> None:
    """Generate a VRAM vs FCR Pareto chart from result files."""
    click.echo("[pareto stub] Implementation in Phase 3.")


@main.command("leaderboard")
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--output-dir", default="leaderboard", show_default=True, type=click.Path())
def leaderboard_cmd(
    results_dir: str,
    output_dir: str,
) -> None:
    """Build leaderboard.{json,csv,md} from a directory of result files."""
    from quantcall.report.leaderboard import build_leaderboard

    board = build_leaderboard(results_dir, output_dir=output_dir)
    click.echo(f"Leaderboard built from {len(board['rows'])} result(s) in {results_dir!r}")
    click.echo(f"  {output_dir}/leaderboard.json")
    click.echo(f"  {output_dir}/leaderboard.csv")
    click.echo(f"  {output_dir}/leaderboard.md")


# ── Suite commands ────────────────────────────────────────────────────────────


@main.group("suite")
def suite_group() -> None:
    """Build and materialize the deterministic evaluation suite."""


@suite_group.command("build")
@click.option("--tiers", default="T0", help="Comma-separated tier list (T0,T1,…)")
@click.option("--sample-size", default=200, show_default=True, help="Instances per tier")
@click.option("--seed", default=42, show_default=True)
@click.option(
    "--bfcl-data-dir",
    default=None,
    type=click.Path(),
    help="Local directory containing BFCL JSONL files (required for T1/T2/T6)",
)
@click.option(
    "--output",
    "output_path",
    default="suite_manifest.json",
    show_default=True,
    type=click.Path(),
)
def suite_build_cmd(
    tiers: str,
    sample_size: int,
    seed: int,
    bfcl_data_dir: str | None,
    output_path: str,
) -> None:
    """Sample N instances per tier and write suite_manifest.json.

    NC/gated tiers (T3/T4) are recorded in license notes but produce no entries.
    T5 (Apache 2.0) also produces no entries yet — adapter not yet implemented.
    BFCL tiers (T1/T2/T6) require --bfcl-data-dir.
    """
    from quantcall.suite.build import build_suite

    tier_list = [t.strip() for t in tiers.split(",") if t.strip()]
    manifest = build_suite(
        tiers=tier_list,
        sample_size=sample_size,
        seed=seed,
        bfcl_data_dir=bfcl_data_dir,
    )
    manifest.save(output_path)
    click.echo(f"Suite manifest written to {output_path}")
    click.echo(f"  tiers         : {tier_list}")
    click.echo(f"  total entries : {len(manifest.entries)}")
    for tier in tier_list:
        n = len(manifest.entries_for_tier(tier))
        license_note = manifest.tier_license_notes.get(tier, "?")[:60]
        click.echo(f"  {tier}: {n} entries  — {license_note}")


@suite_group.command("materialize")
@click.option(
    "--manifest",
    "manifest_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to suite_manifest.json",
)
@click.option(
    "--bfcl-data-dir",
    default=None,
    type=click.Path(),
    help="Local directory containing BFCL JSONL files",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Write reconstructed instances as JSONL (omit to only verify)",
)
def suite_materialize_cmd(
    manifest_path: str,
    bfcl_data_dir: str | None,
    output_path: str | None,
) -> None:
    """Reconstruct NormalizedInstances from a manifest and verify all SHA-256 hashes."""
    import json as _json

    from quantcall.suite.manifest import SuiteManifest
    from quantcall.suite.materialize import materialize_suite

    manifest = SuiteManifest.load(manifest_path)
    instances = materialize_suite(manifest, bfcl_data_dir=bfcl_data_dir)

    if output_path:
        from quantcall.suite.hashes import _instance_to_dict

        lines = [_json.dumps(_instance_to_dict(inst)) for inst in instances]
        Path(output_path).write_text("\n".join(lines) + "\n")
        click.echo(f"Written {len(instances)} instances to {output_path}")
    else:
        click.echo(f"Verified {len(instances)} instances — all hashes match.")


@main.command("validate-config")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def validate_config_cmd(config_path: str) -> None:
    """Validate a QuantCall YAML config file."""
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Config: {config_path}")
    click.echo(f"  backend    : {cfg.backend}")
    click.echo(f"  model      : {cfg.model}")
    click.echo(f"  quant      : {cfg.quant}")
    click.echo(f"  decoding   : {cfg.decoding}")
    click.echo(f"  tiers      : {cfg.tiers}")
    click.echo(f"  sample_size: {cfg.sample_size}")
    click.echo(f"  seed       : {cfg.seed}")
    click.echo("OK")


def _build_backend(cfg: QuantCallConfig) -> Any:
    from quantcall.backends.mock import MockBackend

    if cfg.backend == "mock":
        return MockBackend(model=cfg.model, latency_ms=cfg.mock.latency_ms)
    if cfg.backend == "llama-cpp":
        from quantcall.backends.llama_cpp import LlamaCppBackend  # type: ignore[import]

        return LlamaCppBackend(
            model_path=cfg.model,
            n_ctx=cfg.llama_cpp.n_ctx,
            n_gpu_layers=cfg.llama_cpp.n_gpu_layers,
            max_tokens=cfg.llama_cpp.max_tokens,
            temperature=cfg.temperature,
            chat_format=cfg.llama_cpp.chat_format,
        )
    if cfg.backend == "transformers":
        from quantcall.backends.hf import HFBackend  # type: ignore[import]

        return HFBackend(
            model_id=cfg.model,
            device=cfg.hf.device,
            torch_dtype=cfg.hf.torch_dtype,
            max_new_tokens=cfg.hf.max_new_tokens,
        )
    if cfg.backend == "openai":
        from quantcall.backends.openai_endpoint import OpenAIEndpointBackend  # type: ignore[import]

        return OpenAIEndpointBackend(
            base_url=cfg.openai.base_url,
            model=cfg.model,
            max_tokens=cfg.openai.max_tokens,
            temperature=cfg.temperature,
            timeout_s=cfg.openai.timeout_s,
            api_key_env=cfg.openai.api_key_env,
        )
    if cfg.backend == "vllm":
        from quantcall.backends.vllm_backend import VLLMBackend  # type: ignore[import]

        return VLLMBackend(
            model_id=cfg.model,
            max_new_tokens=cfg.vllm.max_new_tokens,
            temperature=cfg.temperature,
            tensor_parallel_size=cfg.vllm.tensor_parallel_size,
            gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
        )
    raise ValueError(f"Unknown backend: {cfg.backend!r}")


def _load_instances(cfg: QuantCallConfig) -> list[Any]:
    from quantcall.datasets.smoke import load_smoke

    instances: list[Any] = []
    for tier in cfg.tiers:
        if tier == "T0":
            instances.extend(load_smoke())
        elif tier == "T1":
            try:
                from quantcall.datasets.bfcl import load_bfcl  # type: ignore[import]

                instances.extend(
                    load_bfcl(categories=["simple_python", "multiple"], data_dir=cfg.bfcl_data_dir)
                )
            except (ImportError, FileNotFoundError) as exc:
                click.echo(f"T1 requires BFCL data in {cfg.bfcl_data_dir!r}: {exc}", err=True)
        elif tier == "T2":
            try:
                from quantcall.datasets.bfcl import load_bfcl  # type: ignore[import]

                instances.extend(
                    load_bfcl(
                        categories=["parallel", "parallel_multiple"], data_dir=cfg.bfcl_data_dir
                    )
                )
            except (ImportError, FileNotFoundError) as exc:
                click.echo(f"T2 requires BFCL data in {cfg.bfcl_data_dir!r}: {exc}", err=True)
        elif tier == "T3":
            try:
                from quantcall.datasets.toolace import load_toolace  # type: ignore[import]

                instances.extend(load_toolace())
            except ImportError:
                click.echo("T3 requires ToolACE data; skipping.", err=True)
        elif tier == "T4":
            try:
                from quantcall.datasets.xlam import load_xlam  # type: ignore[import]

                instances.extend(load_xlam())
            except ImportError:
                click.echo("T4 requires xlam data; skipping.", err=True)
        elif tier == "T5":
            try:
                from quantcall.datasets.hermes import load_hermes  # type: ignore[import]

                instances.extend(load_hermes())
            except ImportError:
                click.echo("T5 requires Hermes data; skipping.", err=True)
        elif tier == "T6":
            try:
                from quantcall.datasets.bfcl import load_bfcl  # type: ignore[import]

                instances.extend(load_bfcl(categories=["irrelevance"], data_dir=cfg.bfcl_data_dir))
            except (ImportError, FileNotFoundError) as exc:
                click.echo(f"T6 requires BFCL data in {cfg.bfcl_data_dir!r}: {exc}", err=True)
        else:
            click.echo(f"Unknown tier {tier!r}; skipping.", err=True)

    if cfg.sample_size and len(instances) > cfg.sample_size:
        import random

        rng = random.Random(cfg.seed)
        instances = rng.sample(instances, cfg.sample_size)

    return instances

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from quantcall.backends.base import Backend, tools_to_openai_spec
from quantcall.config import QuantCallConfig
from quantcall.datasets.base import NormalizedInstance
from quantcall.manifest import RunManifest, collect_manifest, compute_dataset_sha256
from quantcall.metrics.core import (
    FcrWeights,
    InstanceResult,
    MetricsResult,
    compute_metrics,
    evaluate_instance,
)
from quantcall.parsing.hermes_xml import HermesXmlParser
from quantcall.parsing.raw_json import RawJsonParser


@dataclass
class RunResult:
    config: dict[str, Any]
    metrics: MetricsResult
    manifest: RunManifest
    total_latency_ms: float
    instance_results: list[InstanceResult] = field(default_factory=list)
    peak_vram_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "n": self.metrics.n,
            "svr": self.metrics.svr,
            "tsa": self.metrics.tsa,
            "tsa_precision": self.metrics.tsa_precision,
            "tsa_recall": self.metrics.tsa_recall,
            "ac": self.metrics.ac,
            "abstention": self.metrics.abstention,
            "over_call": self.metrics.over_call,
            "fcr": self.metrics.fcr,
            "total_latency_ms": self.total_latency_ms,
            "vram_gb": (self.peak_vram_mb / 1024.0) if self.peak_vram_mb is not None else None,
            "config": self.config,
            "manifest": asdict(self.manifest),
        }
        return d


def _get_parser(chat_variant: str) -> Any:
    if chat_variant == "qwen3_nothink":
        return HermesXmlParser()
    return RawJsonParser()


def _build_messages(instance: NormalizedInstance, chat_variant: str) -> list[dict[str, Any]]:
    query = instance.query
    if chat_variant == "qwen3_nothink":
        query = f"{query} /no_think"
    return [{"role": "user", "content": query}]


def run_eval(
    cfg: QuantCallConfig,
    instances: list[NormalizedInstance],
    backend: Backend,
    config_path: str | Path = "",
) -> RunResult:
    weights = FcrWeights(
        svr=cfg.metrics.fcr_weights.svr,
        tsa=cfg.metrics.fcr_weights.tsa,
        ac=cfg.metrics.fcr_weights.ac,
        abst=cfg.metrics.fcr_weights.abst,
    )
    parser = _get_parser(cfg.chat_variant)
    instance_results: list[InstanceResult] = []
    total_latency_ms = 0.0
    peak_vram_mb: float | None = None

    for instance in instances:
        messages = _build_messages(instance, cfg.chat_variant)
        tools = tools_to_openai_spec(instance.tools)

        try:
            result = backend.generate_toolcall(messages, tools)
            total_latency_ms += result.latency_ms
            if result.peak_vram_mb is not None:
                peak_vram_mb = max(peak_vram_mb or 0.0, result.peak_vram_mb)
            parsed_calls = parser.parse(result.raw_output)
            parse_succeeded = True
        except Exception:
            parsed_calls = []
            parse_succeeded = False

        inst_result = evaluate_instance(instance, parsed_calls, parse_succeeded)
        instance_results.append(inst_result)

    metrics = compute_metrics(instance_results, weights)
    dataset_sha = compute_dataset_sha256(instances)
    manifest = collect_manifest(config_path, cfg, dataset_sha256=dataset_sha)

    config_dict: dict[str, Any] = {
        "model": cfg.model,
        "backend": cfg.backend,
        "quant": cfg.quant,
        "decoding": cfg.decoding,
        "chat_variant": cfg.chat_variant,
        "tiers": cfg.tiers,
        "sample_size": cfg.sample_size,
        "seed": cfg.seed,
        "temperature": cfg.temperature,
    }

    return RunResult(
        config=config_dict,
        metrics=metrics,
        manifest=manifest,
        total_latency_ms=total_latency_ms,
        instance_results=instance_results,
        peak_vram_mb=peak_vram_mb,
    )


def write_result(result: RunResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result.to_dict(), indent=2) + "\n")

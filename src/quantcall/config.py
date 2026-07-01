from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class FcrWeightsConfig(BaseModel):
    svr: float = Field(default=0.25, ge=0.0, le=1.0)
    tsa: float = Field(default=0.25, ge=0.0, le=1.0)
    ac: float = Field(default=0.25, ge=0.0, le=1.0)
    abst: float = Field(default=0.25, ge=0.0, le=1.0)


class MetricsConfig(BaseModel):
    fcr_weights: FcrWeightsConfig = Field(default_factory=FcrWeightsConfig)


class ReferenceConfig(BaseModel):
    quant: str = "fp16"
    backend: str = "transformers"
    result_file: str | None = None


class MockBackendConfig(BaseModel):
    latency_ms: float = Field(default=5.0, ge=0.0)


class LlamaCppBackendConfig(BaseModel):
    n_ctx: int = Field(default=4096, ge=1)
    n_gpu_layers: int = -1
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    n_threads: int | None = None
    verbose: bool = False
    chat_format: str | None = None


class HFBackendConfig(BaseModel):
    max_new_tokens: int = Field(default=512, ge=1)
    device: str = "cpu"
    torch_dtype: Literal["float32", "float16", "bfloat16", "auto"] = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False


class OpenAIEndpointConfig(BaseModel):
    base_url: str = "http://localhost:8080/v1"
    api_key_env: str | None = None
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    timeout_s: float = Field(default=60.0, gt=0.0)


class VLLMBackendConfig(BaseModel):
    max_new_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    guided_decoding_backend: str = "xgrammar"


class QuantCallConfig(BaseModel):
    model: str = "mock"
    backend: Literal["mock", "llama-cpp", "transformers", "vllm", "openai"] = "mock"
    quant: str = "fp16"
    decoding: Literal["free", "constrained"] = "free"
    # "qwen3_nothink": append "/no_think" to the user turn to suppress Qwen3's
    # <think>...</think> reasoning block, and parse tool calls with the
    # Hermes-style <tool_call>{...}</tool_call> parser instead of raw_json.
    chat_variant: Literal["default", "qwen3_nothink"] = "default"
    tiers: list[str] = Field(default_factory=lambda: ["T0"])
    sample_size: int = Field(default=50, ge=1)
    seed: int = 42
    temperature: float = Field(default=0.0, ge=0.0)
    repeats: int = Field(default=1, ge=1)
    use_gated_xlam: bool = False
    # BFCL: path to local directory with BFCL v4 JSONL files.
    # Defaults to "data/bfcl" relative to CWD if not set.
    bfcl_data_dir: str = "data/bfcl"
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    reference: ReferenceConfig | None = None
    mock: MockBackendConfig = Field(default_factory=MockBackendConfig)
    llama_cpp: LlamaCppBackendConfig = Field(default_factory=LlamaCppBackendConfig)
    hf: HFBackendConfig = Field(default_factory=HFBackendConfig)
    openai: OpenAIEndpointConfig = Field(default_factory=OpenAIEndpointConfig)
    vllm: VLLMBackendConfig = Field(default_factory=VLLMBackendConfig)


def load_config(path: str | Path) -> QuantCallConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return QuantCallConfig.model_validate(data or {})

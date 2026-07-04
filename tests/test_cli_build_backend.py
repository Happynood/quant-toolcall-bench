from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from quantcall.cli import _build_backend
from quantcall.config import QuantCallConfig


def test_build_backend_mock():
    from quantcall.backends.mock import MockBackend

    cfg = QuantCallConfig(backend="mock", model="mock")
    backend = _build_backend(cfg)
    assert isinstance(backend, MockBackend)


def test_build_backend_unknown_raises():
    cfg = QuantCallConfig.model_construct(backend="bogus", model="x")
    with pytest.raises(ValueError, match="Unknown backend"):
        _build_backend(cfg)


def test_build_backend_openai_dispatches_to_openai_endpoint_backend():
    from quantcall.backends.openai_endpoint import OpenAIEndpointBackend

    cfg = QuantCallConfig(backend="openai", model="gpt-test")
    backend = _build_backend(cfg)
    assert isinstance(backend, OpenAIEndpointBackend)
    assert backend.name == "openai"


@pytest.fixture
def _fake_transformers():
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"  # type: ignore[attr-defined]
    fake_torch.float16 = "float16"  # type: ignore[attr-defined]
    fake_torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0

    model = MagicMock()
    model.to.return_value = model

    tokenizer_cls = MagicMock()
    tokenizer_cls.from_pretrained.return_value = tokenizer
    model_cls = MagicMock()
    model_cls.from_pretrained.return_value = model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = tokenizer_cls  # type: ignore[attr-defined]
    fake_transformers.AutoModelForCausalLM = model_cls  # type: ignore[attr-defined]
    fake_transformers.BitsAndBytesConfig = MagicMock()  # type: ignore[attr-defined]

    sys.modules["torch"] = fake_torch
    sys.modules["transformers"] = fake_transformers
    yield
    sys.modules.pop("torch", None)
    sys.modules.pop("transformers", None)
    sys.modules.pop("quantcall.backends.hf", None)


def test_build_backend_transformers_dispatches_to_hf_backend(_fake_transformers):
    from quantcall.backends.hf import HFBackend

    cfg = QuantCallConfig(backend="transformers", model="fake/model")
    backend = _build_backend(cfg)
    assert isinstance(backend, HFBackend)
    assert backend.name == "transformers"


@pytest.fixture
def _fake_vllm():
    llm_instance = MagicMock()
    llm_cls = MagicMock(return_value=llm_instance)
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = llm_cls  # type: ignore[attr-defined]
    fake_vllm.SamplingParams = MagicMock()  # type: ignore[attr-defined]
    sys.modules["vllm"] = fake_vllm
    yield
    sys.modules.pop("vllm", None)
    sys.modules.pop("quantcall.backends.vllm_backend", None)


def test_build_backend_vllm_dispatches_to_vllm_backend(_fake_vllm):
    from quantcall.backends.vllm_backend import VLLMBackend

    cfg = QuantCallConfig(backend="vllm", model="fake/model")
    backend = _build_backend(cfg)
    assert isinstance(backend, VLLMBackend)
    assert backend.name == "vllm"

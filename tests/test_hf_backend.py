from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def to(self, device: Any) -> _FakeTensor:
        return self

    def __getitem__(self, item: Any) -> _FakeTensor:
        return _FakeTensor((1,))


class _FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


def _install_fake_transformers(generated_text: str = "hello world") -> dict[str, MagicMock]:
    """Install fake `torch` + `transformers` modules into sys.modules.

    Returns the mock objects so tests can assert on call args. Real
    transformers/torch are never required to run these tests.
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"  # type: ignore[attr-defined]
    fake_torch.float16 = "float16"  # type: ignore[attr-defined]
    fake_torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    fake_torch.no_grad = lambda: _FakeNoGrad()  # type: ignore[attr-defined]

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.apply_chat_template.return_value = "<rendered prompt>"

    input_ids = _FakeTensor((1, 5))

    def _tokenizer_call(*args: Any, **kwargs: Any) -> dict[str, _FakeTensor]:
        return {"input_ids": input_ids}

    tokenizer.side_effect = _tokenizer_call

    output_ids = MagicMock()
    output_ids.shape = (1, 5 + 3)
    output_ids.__getitem__.return_value = "decoded-ids"

    model = MagicMock()
    model.device = "cpu"
    model.generate.return_value = output_ids
    model.to.return_value = model
    model.eval.return_value = None

    tokenizer_cls = MagicMock()
    tokenizer_cls.from_pretrained.return_value = tokenizer

    model_cls = MagicMock()
    model_cls.from_pretrained.return_value = model

    tokenizer.decode.return_value = generated_text

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = tokenizer_cls  # type: ignore[attr-defined]
    fake_transformers.AutoModelForCausalLM = model_cls  # type: ignore[attr-defined]
    fake_transformers.BitsAndBytesConfig = MagicMock()  # type: ignore[attr-defined]

    sys.modules["torch"] = fake_torch
    sys.modules["transformers"] = fake_transformers

    return {
        "tokenizer_cls": tokenizer_cls,
        "tokenizer": tokenizer,
        "model_cls": model_cls,
        "model": model,
    }


@pytest.fixture(autouse=True)
def _cleanup_fake_modules():
    yield
    sys.modules.pop("torch", None)
    sys.modules.pop("transformers", None)
    sys.modules.pop("quantcall.backends.hf", None)


def _sample_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]


def test_hf_backend_name():
    _install_fake_transformers()
    from quantcall.backends.hf import HFBackend

    backend = HFBackend(model_id="fake/model")
    assert backend.name == "transformers"


def test_hf_backend_uses_apply_chat_template_with_tools():
    mocks = _install_fake_transformers()
    from quantcall.backends.hf import HFBackend

    backend = HFBackend(model_id="fake/model")
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    tools = _sample_tools()
    backend.generate_toolcall(messages, tools)

    mocks["tokenizer"].apply_chat_template.assert_called_once()
    _, kwargs = mocks["tokenizer"].apply_chat_template.call_args
    assert kwargs["tools"] == tools
    assert kwargs["add_generation_prompt"] is True


def test_hf_backend_returns_tool_call_result_with_expected_fields():
    _install_fake_transformers(generated_text='{"name": "get_weather", "arguments": {}}')
    from quantcall.backends.hf import HFBackend

    backend = HFBackend(model_id="fake/model")
    result = backend.generate_toolcall([{"role": "user", "content": "weather?"}], _sample_tools())

    assert result.raw_output == '{"name": "get_weather", "arguments": {}}'
    assert result.input_tokens == 5
    assert result.output_tokens == 3
    assert result.latency_ms >= 0
    assert result.tokens_per_second is None or result.tokens_per_second > 0


def test_hf_backend_passes_dtype_and_quantization_to_from_pretrained():
    mocks = _install_fake_transformers()
    from quantcall.backends.hf import HFBackend

    HFBackend(model_id="fake/model", torch_dtype="bfloat16", load_in_4bit=True, device="cuda:0")

    _, kwargs = mocks["model_cls"].from_pretrained.call_args
    assert kwargs["torch_dtype"] == "bfloat16"
    assert kwargs["quantization_config"] is not None
    assert kwargs["device_map"] == "cuda:0"

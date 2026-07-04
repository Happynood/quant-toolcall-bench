from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest


def _install_fake_vllm(generated_text: str = '{"name": "get_weather", "arguments": {}}'):
    completion = MagicMock()
    completion.text = generated_text
    completion.token_ids = [1, 2, 3]

    chat_output = MagicMock()
    chat_output.outputs = [completion]
    chat_output.prompt_token_ids = [1, 2, 3, 4, 5]

    llm_instance = MagicMock()
    llm_instance.chat.return_value = [chat_output]

    llm_cls = MagicMock(return_value=llm_instance)
    sampling_params_cls = MagicMock()

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = llm_cls  # type: ignore[attr-defined]
    fake_vllm.SamplingParams = sampling_params_cls  # type: ignore[attr-defined]

    sys.modules["vllm"] = fake_vllm

    return {
        "llm_cls": llm_cls,
        "llm_instance": llm_instance,
        "sampling_params_cls": sampling_params_cls,
        "chat_output": chat_output,
    }


@pytest.fixture(autouse=True)
def _cleanup_fake_modules():
    yield
    sys.modules.pop("vllm", None)
    sys.modules.pop("quantcall.backends.vllm_backend", None)


def _sample_tools() -> list[dict[str, Any]]:
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


def test_vllm_backend_name():
    _install_fake_vllm()
    from quantcall.backends.vllm_backend import VLLMBackend

    backend = VLLMBackend(model_id="fake/model")
    assert backend.name == "vllm"


def test_vllm_backend_passes_model_and_engine_kwargs_to_llm():
    mocks = _install_fake_vllm()
    from quantcall.backends.vllm_backend import VLLMBackend

    VLLMBackend(
        model_id="fake/model",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.5,
        dtype="float16",
    )

    _, kwargs = mocks["llm_cls"].call_args
    assert kwargs["model"] == "fake/model"
    assert kwargs["tensor_parallel_size"] == 2
    assert kwargs["gpu_memory_utilization"] == 0.5
    assert kwargs["dtype"] == "float16"


def test_vllm_backend_calls_chat_with_messages_and_tools():
    mocks = _install_fake_vllm()
    from quantcall.backends.vllm_backend import VLLMBackend

    backend = VLLMBackend(model_id="fake/model")
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    tools = _sample_tools()
    backend.generate_toolcall(messages, tools)

    args, kwargs = mocks["llm_instance"].chat.call_args
    assert args[0] == messages
    assert kwargs["tools"] == tools


def test_vllm_backend_omits_tools_when_none_given():
    mocks = _install_fake_vllm()
    from quantcall.backends.vllm_backend import VLLMBackend

    backend = VLLMBackend(model_id="fake/model")
    backend.generate_toolcall([{"role": "user", "content": "hi"}], [])

    _, kwargs = mocks["llm_instance"].chat.call_args
    assert kwargs["tools"] is None


def test_vllm_backend_returns_tool_call_result_with_expected_fields():
    _install_fake_vllm(generated_text='{"name": "get_weather", "arguments": {"city": "Paris"}}')
    from quantcall.backends.vllm_backend import VLLMBackend

    backend = VLLMBackend(model_id="fake/model")
    result = backend.generate_toolcall([{"role": "user", "content": "weather?"}], _sample_tools())

    assert result.raw_output == '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    assert result.input_tokens == 5
    assert result.output_tokens == 3
    assert result.latency_ms >= 0
    assert result.tokens_per_second is None or result.tokens_per_second > 0

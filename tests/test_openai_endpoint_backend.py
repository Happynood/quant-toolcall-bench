"""Unit tests for the OpenAI-compatible endpoint backend.

All tests mock urllib's urlopen -- no real network call is made. A separate
integration test (marked pytest.mark.integration) hits a real running
OpenAI-compatible server when OPENAI_ENDPOINT_BASE_URL is set.
"""

from __future__ import annotations

import json
import os
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

import quantcall.backends.openai_endpoint as openai_endpoint_mod
from quantcall.backends.base import ToolCallResult
from quantcall.backends.openai_endpoint import OpenAIEndpointBackend

_BASE_URL = "http://localhost:8080/v1"
_MODEL = "test-model"


def _make_chat_response(
    content: str | None = "mocked completion",
    tool_calls: list[dict] | None = None,
    prompt_tokens: int = 5,
    completion_tokens: int = 10,
) -> dict:
    message: dict = {"role": "assistant"}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
        message["content"] = None
    else:
        message["content"] = content
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _make_mock_urlopen(body: dict) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(body).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return MagicMock(return_value=response)


def _messages() -> list[dict]:
    return [{"role": "user", "content": "What is the weather in Paris?"}]


def _tools() -> list[dict]:
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


def test_backend_name_is_openai():
    backend = OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
    assert backend.name == "openai"


def test_generate_toolcall_returns_tool_call_result():
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        result = backend.generate_toolcall(_messages(), [])
        assert isinstance(result, ToolCallResult)


def test_generate_toolcall_uses_content_when_no_tool_calls():
    mock_urlopen = _make_mock_urlopen(_make_chat_response(content="no call needed"))
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        result = backend.generate_toolcall(_messages(), _tools())
        assert result.raw_output == "no call needed"


def test_generate_toolcall_parses_native_tool_calls_to_json():
    tool_calls = [
        {
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Paris"}),
            }
        }
    ]
    mock_urlopen = _make_mock_urlopen(_make_chat_response(tool_calls=tool_calls))
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        result = backend.generate_toolcall(_messages(), _tools())
        parsed = json.loads(result.raw_output)
        assert parsed["name"] == "get_weather"
        assert parsed["arguments"] == {"city": "Paris"}


def test_generate_toolcall_uses_usage_tokens():
    mock_urlopen = _make_mock_urlopen(_make_chat_response(prompt_tokens=7, completion_tokens=13))
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        result = backend.generate_toolcall(_messages(), [])
        assert result.input_tokens == 7
        assert result.output_tokens == 13


def test_generate_toolcall_posts_to_chat_completions_path():
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        backend.generate_toolcall(_messages(), [])
        request = mock_urlopen.call_args.args[0]
        assert request.full_url == "http://localhost:8080/v1/chat/completions"


def test_generate_toolcall_strips_trailing_slash_from_base_url():
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = OpenAIEndpointBackend(base_url="http://localhost:8080/v1/", model=_MODEL)
        backend.generate_toolcall(_messages(), [])
        request = mock_urlopen.call_args.args[0]
        assert request.full_url == "http://localhost:8080/v1/chat/completions"


def test_generate_toolcall_sends_tools_and_tool_choice_when_tools_present():
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL, max_tokens=42)
        backend.generate_toolcall(_messages(), _tools())
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == _MODEL
        assert payload["max_tokens"] == 42
        assert payload["tools"] == _tools()
        assert payload["tool_choice"] == "auto"
        assert payload["messages"] == _messages()


def test_generate_toolcall_omits_tools_when_none_given():
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        backend.generate_toolcall(_messages(), [])
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        assert "tools" not in payload


def test_generate_toolcall_omits_authorization_header_by_default():
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        backend.generate_toolcall(_messages(), [])
        request = mock_urlopen.call_args.args[0]
        assert request.get_header("Authorization") is None


def test_generate_toolcall_sends_authorization_header_when_api_key_env_set():
    mock_urlopen = _make_mock_urlopen(_make_chat_response())
    with (
        patch.object(openai_endpoint_mod, "urlopen", mock_urlopen),
        patch.dict(os.environ, {"OPENAI_TEST_KEY": "secret-value-123"}),
    ):
        backend = OpenAIEndpointBackend(
            base_url=_BASE_URL, model=_MODEL, api_key_env="OPENAI_TEST_KEY"
        )
        backend.generate_toolcall(_messages(), [])
        request = mock_urlopen.call_args.args[0]
        assert request.get_header("Authorization") == "Bearer secret-value-123"


def test_generate_toolcall_raises_runtime_error_on_http_error():
    mock_urlopen = MagicMock(
        side_effect=urllib.error.HTTPError(
            url=_BASE_URL, code=500, msg="Internal Server Error", hdrs=MagicMock(), fp=None
        )
    )
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        with pytest.raises(RuntimeError, match="OpenAI-compatible endpoint request failed"):
            backend.generate_toolcall(_messages(), [])


def test_generate_toolcall_raises_runtime_error_on_connection_error():
    mock_urlopen = MagicMock(side_effect=urllib.error.URLError("connection refused"))
    with patch.object(openai_endpoint_mod, "urlopen", mock_urlopen):
        backend = OpenAIEndpointBackend(base_url=_BASE_URL, model=_MODEL)
        with pytest.raises(RuntimeError, match="OpenAI-compatible endpoint request failed"):
            backend.generate_toolcall(_messages(), [])


@pytest.mark.integration
def test_integration_real_endpoint():
    """Real HTTP call against a live OpenAI-compatible server.

    Set OPENAI_ENDPOINT_BASE_URL (e.g. a local llama_cpp.server instance) to
    run this for real; skipped otherwise.
    """
    base_url = os.environ.get("OPENAI_ENDPOINT_BASE_URL")
    if not base_url:
        pytest.skip("OPENAI_ENDPOINT_BASE_URL env var not set -- skipping live endpoint test")

    model = os.environ.get("OPENAI_ENDPOINT_MODEL", "default")
    backend = OpenAIEndpointBackend(base_url=base_url, model=model, max_tokens=16)
    result = backend.generate_toolcall(
        [{"role": "user", "content": "What is the weather in Paris?"}],
        _tools(),
    )
    assert result.latency_ms > 0
    assert isinstance(result.raw_output, str)

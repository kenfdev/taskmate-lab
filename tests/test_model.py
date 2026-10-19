import asyncio
import json
from copy import deepcopy
from typing import Any

import httpx
import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.tools import tool

from taskmate.model import ChatGPTResponsesModel, RESPONSES_OUTPUT_KEY, ResponsesError

pytestmark = pytest.mark.asyncio

MODEL = "test-account-model"
TOKEN = "fake-oauth-token"
CATALOG = {"models": [{"slug": MODEL, "display_name": "Test", "visibility": "list"}]}
FUNCTION = {
    "type": "function",
    "function": {
        "name": "read_note",
        "description": "Read a local note.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


async def token_getter() -> str:
    return TOKEN


def output_text(text: str = "done") -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_test",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def function_call(**updates: Any) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": "fc_test",
        "status": "completed",
        "namespace": "taskmate",
        "name": "read_note",
        "call_id": "call_test",
        "arguments": '{"path":"notes/a.txt"}',
        **updates,
    }


def completed(*items: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_test",
            "model": MODEL,
            "status": "completed",
            "output": list(items),
            "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        },
    }


def event_bytes(*events: dict[str, Any]) -> bytes:
    return "".join(
        "event: " + event["type"] + "\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        for event in events
    ).encode()


def sse(*events: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        content=event_bytes(*events),
    )


class Wire:
    def __init__(self, *responses: httpx.Response, catalog: Any = None) -> None:
        self.responses = list(responses)
        self.catalog = deepcopy(CATALOG if catalog is None else catalog)
        self.requests: list[httpx.Request] = []
        self.posts: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.scheme == "https"
        assert request.url.host == "api.openai.com"
        assert request.headers["authorization"] == "Bearer " + TOKEN
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json=self.catalog)
        assert request.method == "POST"
        assert request.url.path == "/v1/responses"
        self.posts.append(json.loads(request.content))
        return self.responses.pop(0)


def adapter(client: httpx.AsyncClient, model: str | None = MODEL) -> ChatGPTResponsesModel:
    return ChatGPTResponsesModel(
        model=model, access_token_getter=token_getter, http_client=client
    )


async def test_catalog_is_account_specific_and_preserves_server_order() -> None:
    wire = Wire(
        catalog={
            "models": [
                {"slug": "z", "visibility": "list"},
                {"slug": "hidden", "visibility": "hide"},
                {"slug": "a", "visibility": "list"},
            ]
        }
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        model = adapter(client, None)
        assert await model.available_models() == ["z", "a"]
        with pytest.raises(ValueError, match="Select a model"):
            await model.ainvoke("hello")
    assert len(wire.requests) == 1
    assert wire.posts == []


@pytest.mark.parametrize("catalog", [{}, {"models": {}}, {"models": [None]}, {"models": [{"visibility": "list"}]}])
async def test_invalid_catalog_is_rejected(catalog: Any) -> None:
    wire = Wire(catalog=catalog)
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        with pytest.raises(ResponsesError):
            await adapter(client).available_models()


async def test_request_uses_oauth_public_responses_and_full_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-never-be-used")
    wire = Wire(sse(completed(output_text("hello"))))
    messages = [
        SystemMessage("Be concise."),
        HumanMessage("first"),
        AIMessage("previous answer"),
        HumanMessage("second"),
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        result = await adapter(client).ainvoke(messages, reasoning={"effort": "low"})
    assert result.content == "hello"
    assert result.usage_metadata == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    assert wire.posts == [
        {
            "model": MODEL,
            "input": [
                {"role": "developer", "content": "Be concise."},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "previous answer"},
                {"role": "user", "content": "second"},
            ],
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "low"},
        }
    ]
    assert [request.method for request in wire.requests] == ["GET", "POST"]


async def test_namespaced_tool_and_completed_reasoning_roundtrip_after_serialization() -> None:
    reasoning = {
        "id": "rs_test",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "Checked the request."}],
        "encrypted_content": "complete-opaque-test-value",
        "status": "completed",
    }
    terminal_reasoning = {**reasoning, "encrypted_content": "incomplete-test-value"}
    call = function_call()
    wire = Wire(
        sse(
            {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
            {"type": "response.output_item.done", "output_index": 1, "item": call},
            completed(terminal_reasoning, call),
        ),
        sse(completed(output_text())),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        bound = adapter(client).bind_tools([FUNCTION], tool_choice="any")
        first = await bound.ainvoke([HumanMessage("Read the note.")])
        assert first.tool_calls == [
            {
                "name": "read_note",
                "args": {"path": "notes/a.txt"},
                "id": "call_test",
                "type": "tool_call",
            }
        ]
        assert first.content == ""
        assert first.additional_kwargs[RESPONSES_OUTPUT_KEY] == [reasoning, call]
        saved = messages_from_dict(messages_to_dict([first]))[0]
        second = await bound.ainvoke(
            [
                HumanMessage("Read the note."),
                saved,
                ToolMessage("The note says hello.", tool_call_id="call_test"),
            ]
        )
    assert second.content == "done"
    assert wire.posts[0]["tools"] == [
        {
            "type": "namespace",
            "name": "taskmate",
            "description": "Tools executed by the local Taskmate agent.",
            "tools": [
                {
                    "type": "function",
                    **FUNCTION["function"],
                }
            ],
        }
    ]
    assert wire.posts[0]["tool_choice"] == "required"
    assert wire.posts[1]["input"] == [
        {"role": "user", "content": "Read the note."},
        reasoning,
        call,
        {"type": "function_call_output", "call_id": "call_test", "output": "The note says hello."},
    ]
    assert "previous_response_id" not in wire.posts[1]
    assert wire.posts[1]["store"] is False
    assert wire.posts[1]["stream"] is True


async def test_deepagents_argument_trimming_is_not_undone_by_saved_wire_items() -> None:
    wire = Wire(sse(completed(function_call())), sse(completed(output_text())))
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        bound = adapter(client).bind_tools([FUNCTION])
        first = await bound.ainvoke("Read.")
        first.tool_calls[0]["args"] = {"path": "[trimmed]"}
        await bound.ainvoke(
            [HumanMessage("Read."), first, ToolMessage("hello", tool_call_id="call_test")]
        )
    assert json.loads(wire.posts[1]["input"][1]["arguments"]) == {"path": "[trimmed]"}


async def test_custom_tool_input_and_output_roundtrip() -> None:
    custom = {
        "type": "custom_tool_call",
        "id": "ct_test",
        "namespace": "taskmate",
        "name": "echo",
        "call_id": "call_custom",
        "input": "free form input",
    }
    wire = Wire(sse(completed(custom)), sse(completed(output_text())))
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        bound = adapter(client).bind_tools(
            [{"type": "custom", "name": "echo", "description": "Echo text."}]
        )
        first = await bound.ainvoke("Echo.")
        assert first.tool_calls[0]["args"] == {"__arg1": "free form input"}
        await bound.ainvoke(
            [HumanMessage("Echo."), first, ToolMessage("echoed", tool_call_id="call_custom")]
        )
    assert wire.posts[1]["input"][-1] == {
        "type": "custom_tool_call_output",
        "call_id": "call_custom",
        "output": "echoed",
    }


async def test_langchain_tool_binding_does_not_mutate_base_model() -> None:
    @tool
    def read_note(path: str) -> str:
        """Read a local note."""
        return path

    wire = Wire(sse(completed(output_text())), sse(completed(output_text())))
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        base = adapter(client)
        bound = base.bind_tools([read_note])
        await bound.ainvoke("hello")
        await base.ainvoke("hello")
    assert wire.posts[0]["tools"][0]["tools"][0]["name"] == "read_note"
    assert "tools" not in wire.posts[1]


@pytest.mark.parametrize(
    "changes",
    [
        {"namespace": "other"},
        {"name": "other.read_note", "namespace": None},
        {"name": "unbound"},
        {"arguments": "{broken"},
        {"arguments": "[]"},
    ],
)
async def test_unsafe_or_malformed_tool_calls_are_rejected(changes: dict[str, Any]) -> None:
    wire = Wire(sse(completed(function_call(**changes))))
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        with pytest.raises(ResponsesError):
            await adapter(client).bind_tools([FUNCTION]).ainvoke("hello")


async def test_explicit_namespace_prefix_without_namespace_field_is_resolved() -> None:
    call = function_call(name="taskmate.read_note")
    del call["namespace"]
    wire = Wire(sse(completed(call)))
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        result = await adapter(client).bind_tools([FUNCTION]).ainvoke("hello")
    assert result.tool_calls[0]["name"] == "read_note"


@pytest.mark.parametrize(
    "options",
    [
        {"temperature": 0},
        {"store": True},
        {"stream": False},
        {"previous_response_id": "resp_old"},
        {"max_output_tokens": 10},
        {"metadata": {}},
        {"background": True},
        {"stop": []},
        {"tool_choice": {"type": "function", "name": "read_note"}},
    ],
)
async def test_unsupported_options_fail_before_network_io(options: dict[str, Any]) -> None:
    wire = Wire()
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        with pytest.raises(ValueError):
            await adapter(client).ainvoke("hello", **options)
    assert wire.requests == []


async def test_selected_model_must_be_in_current_account_catalog() -> None:
    wire = Wire()
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        with pytest.raises(ValueError, match="catalog"):
            await adapter(client, "unavailable-model").ainvoke("hello")
    assert len(wire.requests) == 1
    assert wire.posts == []


@pytest.mark.parametrize("status", [401, 403, 429, 500, 302])
async def test_http_failure_is_sanitized_and_not_retried(status: int) -> None:
    wire = Wire(
        httpx.Response(
            status,
            headers={"location": "https://untrusted.example/"},
            text="server-detail-that-must-not-escape",
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        with pytest.raises(ResponsesError) as caught:
            await adapter(client).ainvoke("hello")
    assert caught.value.status_code == status
    assert "server-detail" not in str(caught.value)
    assert TOKEN not in str(caught.value)
    assert len(wire.posts) == 1
    assert len(wire.requests) == 2


@pytest.mark.parametrize(
    "kind", ["error", "response.failed", "response.incomplete", "response.cancelled"]
)
async def test_unsuccessful_stream_events_are_errors(kind: str) -> None:
    wire = Wire(sse({"type": kind, "message": "private-provider-detail"}))
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        with pytest.raises(ResponsesError) as caught:
            await adapter(client).ainvoke("hello")
    assert "private-provider-detail" not in str(caught.value)


@pytest.mark.parametrize(
    "data",
    [
        b"data: {broken}\n\n",
        b"data: {}\n\n",
        b"data: [DONE]\n\n",
        b'data: {"type":"response.created"}\n\n',
        b'data: {"type":"response.created"}',
    ],
)
async def test_malformed_or_truncated_stream_is_not_returned_as_success(data: bytes) -> None:
    wire = Wire(httpx.Response(200, headers={"content-type": "text/event-stream"}, content=data))
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        with pytest.raises(ResponsesError):
            await adapter(client).ainvoke("hello")


async def test_non_sse_response_is_rejected() -> None:
    wire = Wire(httpx.Response(200, json={"output": []}))
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        with pytest.raises(ResponsesError, match="event stream"):
            await adapter(client).ainvoke("hello")


class ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.closed = False

    async def __aiter__(self):
        for offset in range(0, len(self.data), 3):
            yield self.data[offset : offset + 3]

    async def aclose(self) -> None:
        self.closed = True


async def test_sse_handles_network_chunk_boundaries_and_unicode() -> None:
    stream = ChunkedStream(b": keepalive\n\n" + event_bytes(completed(output_text("こんにちは"))))
    wire = Wire(
        httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        result = await adapter(client).ainvoke("hello")
    assert result.content == "こんにちは"
    assert stream.closed


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield b": connected\n\n"
        self.started.set()
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


async def test_cancellation_propagates_and_closes_stream() -> None:
    stream = BlockingStream()
    wire = Wire(
        httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        task = asyncio.create_task(adapter(client).ainvoke("hello"))
        await asyncio.wait_for(stream.started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
        assert not client.is_closed


async def test_orphan_tool_output_and_multimodal_input_fail_before_network_io() -> None:
    wire = Wire()
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        model = adapter(client)
        with pytest.raises(ValueError):
            await model.ainvoke([ToolMessage("orphan", tool_call_id="missing")])
        with pytest.raises(ValueError, match="text"):
            await model.ainvoke(
                [HumanMessage(content=[{"type": "image_url", "image_url": {"url": "test"}}])]
            )
    assert wire.requests == []


async def test_sync_invocation_is_explicitly_unsupported() -> None:
    wire = Wire()
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        with pytest.raises(NotImplementedError, match="ainvoke"):
            adapter(client).invoke("hello")
    assert wire.requests == []


async def test_credentials_are_not_serialized_with_model_configuration() -> None:
    wire = Wire()
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        model = adapter(client)
        config = model.model_dump()
        assert "access_token_getter" not in config
        assert "http_client" not in config
        assert TOKEN not in repr(model)


async def test_token_counting_is_local_and_includes_tools() -> None:
    wire = Wire()
    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
        model = adapter(client)
        messages = [HumanMessage("hello")]
        assert model.get_num_tokens("abcdefgh") == 2
        assert model.get_num_tokens("") == 0
        assert model.get_num_tokens_from_messages(messages, tools=[FUNCTION]) > (
            model.get_num_tokens_from_messages(messages)
        )
    assert wire.requests == []

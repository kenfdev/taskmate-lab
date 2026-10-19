"""Async LangChain adapter for ChatGPT plan usage through public Responses."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from copy import deepcopy
from typing import Any, Literal

import httpx
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

AccessTokenGetter = Callable[[], Awaitable[str]]
JsonObject = dict[str, Any]
RESPONSES_OUTPUT_KEY = "taskmate_responses_output"
_MODELS_URL = "https://api.openai.com/v1/models"
_RESPONSES_URL = "https://api.openai.com/v1/responses"
_NAMESPACE = "taskmate"
_ALLOWED_OPTIONS = frozenset(
    {"instructions", "reasoning", "text", "parallel_tool_calls", "tool_choice", "tools"}
)
_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_OUTPUT_TYPES = frozenset({"message", "reasoning", "function_call", "custom_tool_call"})
_MAX_SSE_EVENT_CHARS = 16 * 1024 * 1024


class ResponsesError(RuntimeError):
    """A sanitized transport or protocol failure without request credentials."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _check_status(response: httpx.Response) -> None:
    if 200 <= response.status_code < 300:
        return
    messages = {
        401: "ChatGPT sign-in expired or was revoked. Sign in again.",
        403: "This ChatGPT account cannot access the requested resource.",
        429: "The ChatGPT account reached a usage or rate limit.",
    }
    raise ResponsesError(
        messages.get(response.status_code, "The OpenAI request failed."),
        status_code=response.status_code,
    )


def _text(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif (
            isinstance(block, dict)
            and block.get("type") in {"text", "input_text", "output_text"}
            and isinstance(block.get("text"), str)
        ):
            parts.append(block["text"])
        else:
            raise ValueError("This PoC accepts text message content only.")
    return "".join(parts)


def _tool_specs(tools: Sequence[Any], strict: bool | None = None) -> list[JsonObject]:
    result: list[JsonObject] = []
    names: set[str] = set()
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "custom":
            converted = deepcopy(tool)
        else:
            converted = deepcopy(convert_to_openai_tool(tool, strict=strict))
        kind = converted.get("type")
        if kind not in {"function", "custom"}:
            raise ValueError("Only locally executed function and custom tools are supported.")
        definition = converted.get("function", converted)
        name = definition.get("name")
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name) or name in names:
            raise ValueError("Tool names must be unique simple identifiers of at most 64 characters.")
        names.add(name)
        keys = (
            {"name", "description", "parameters", "strict"}
            if kind == "function"
            else {"name", "description", "format"}
        )
        spec = {"type": kind, **{key: value for key, value in definition.items() if key in keys}}
        if kind == "function":
            spec.setdefault("parameters", {"type": "object", "properties": {}})
            if strict is not None:
                spec["strict"] = strict
        result.append(spec)
    return result


def _history(messages: Sequence[BaseMessage]) -> list[JsonObject]:
    result: list[JsonObject] = []
    call_kinds: dict[str, str] = {}
    answered: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            call_id = message.tool_call_id
            if call_id not in call_kinds or call_id in answered:
                raise ValueError("A tool result must match one preceding, unanswered tool call.")
            answered.add(call_id)
            result.append(
                {
                    "type": call_kinds[call_id] + "_output",
                    "call_id": call_id,
                    "output": _text(message.content),
                }
            )
            continue
        if isinstance(message, AIMessage):
            saved = message.additional_kwargs.get(RESPONSES_OUTPUT_KEY)
            if saved is not None:
                if not isinstance(saved, list) or any(
                    not isinstance(item, dict) or item.get("type") not in _OUTPUT_TYPES
                    for item in saved
                ):
                    raise ValueError("The saved Responses output is invalid.")
                items = deepcopy(saved)
                calls_by_id = {call["id"]: call for call in message.tool_calls}
                for item in items:
                    if item.get("type") not in {"function_call", "custom_tool_call"}:
                        continue
                    call = calls_by_id.pop(item.get("call_id"), None)
                    if call is None:
                        raise ValueError("Saved Responses calls do not match the message tool calls.")
                    if item["type"] == "function_call":
                        original_args = json.loads(item["arguments"])
                        if call["args"] != original_args:
                            item["arguments"] = json.dumps(call["args"], ensure_ascii=False)
                    elif call["args"] != {"__arg1": item["input"]}:
                        value = call["args"].get("__arg1")
                        if not isinstance(value, str):
                            raise ValueError("Custom tool input must remain a string.")
                        item["input"] = value
                if calls_by_id:
                    raise ValueError("Saved Responses calls do not match the message tool calls.")
            else:
                items = []
                text = _text(message.content)
                if text:
                    items.append({"role": "assistant", "content": text})
                for call in message.tool_calls:
                    if not call.get("id"):
                        raise ValueError("Every tool call needs its original call ID.")
                    items.append(
                        {
                            "type": "function_call",
                            "namespace": _NAMESPACE,
                            "name": call["name"],
                            "call_id": call["id"],
                            "arguments": json.dumps(call["args"], ensure_ascii=False),
                        }
                    )
            for item in items:
                if item.get("type") in {"function_call", "custom_tool_call"}:
                    call_id = item.get("call_id")
                    if not isinstance(call_id, str) or not call_id or call_id in call_kinds:
                        raise ValueError("Tool call IDs must be nonempty and unique.")
                    call_kinds[call_id] = item["type"]
            result.extend(items)
            continue
        if isinstance(message, SystemMessage):
            role = "developer"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, ChatMessage) and message.role in {"system", "developer", "user"}:
            role = "developer" if message.role == "system" else message.role
        else:
            raise ValueError("Unsupported message role.")
        result.append({"role": role, "content": _text(message.content)})
    return result


async def _events(response: httpx.Response) -> AsyncIterator[JsonObject]:
    lines: list[str] = []
    size = 0
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            size += len(value)
            if size > _MAX_SSE_EVENT_CHARS:
                raise ResponsesError("The inference stream exceeded its event-size limit.")
            lines.append(value)
        elif not line and lines:
            data = "\n".join(lines)
            lines = []
            size = 0
            if data == "[DONE]":
                return
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                raise ResponsesError("The inference stream contained malformed JSON.") from None
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise ResponsesError("The inference stream contained an invalid event.")
            yield event
    if lines:
        raise ResponsesError("The inference stream ended inside an event.")


def _tool_name(item: JsonObject, kinds: dict[str, str]) -> str:
    namespace = item.get("namespace")
    name = item.get("name")
    if namespace not in {None, _NAMESPACE} or not isinstance(name, str):
        raise ResponsesError("The model returned a tool outside the bound namespace.")
    if namespace is None and name.startswith(_NAMESPACE + "."):
        name = name[len(_NAMESPACE) + 1 :]
    expected_kind = "function" if item["type"] == "function_call" else "custom"
    if kinds.get(name) != expected_kind:
        raise ResponsesError("The model returned an unbound tool.")
    return name


def _message(response: JsonObject, kinds: dict[str, str]) -> AIMessage:
    output = response.get("output")
    if not isinstance(output, list):
        raise ResponsesError("The completed response is missing its output items.")
    text: list[str] = []
    calls: list[JsonObject] = []
    seen: set[str] = set()
    for item in output:
        if not isinstance(item, dict) or item.get("type") not in _OUTPUT_TYPES:
            raise ResponsesError("The response contains an unsupported output item.")
        kind = item["type"]
        if kind == "message":
            content = item.get("content")
            if not isinstance(content, list):
                raise ResponsesError("The response contains invalid message content.")
            for block in content:
                if not isinstance(block, dict):
                    raise ResponsesError("The response contains invalid message content.")
                field = {"output_text": "text", "refusal": "refusal"}.get(block.get("type"))
                if field is None or not isinstance(block.get(field), str):
                    raise ResponsesError("The response contains unsupported message content.")
                text.append(block[field])
        elif kind in {"function_call", "custom_tool_call"}:
            name = _tool_name(item, kinds)
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id or call_id in seen:
                raise ResponsesError("The response contains an invalid tool call ID.")
            seen.add(call_id)
            if kind == "function_call":
                try:
                    args = json.loads(item["arguments"])
                except (KeyError, TypeError, json.JSONDecodeError):
                    raise ResponsesError("The model returned invalid tool arguments.") from None
                if not isinstance(args, dict):
                    raise ResponsesError("Function arguments must be a JSON object.")
            else:
                if not isinstance(item.get("input"), str):
                    raise ResponsesError("The model returned invalid custom tool input.")
                args = {"__arg1": item["input"]}
            calls.append({"name": name, "args": args, "id": call_id, "type": "tool_call"})
    usage = response.get("usage")
    usage_metadata = None
    if isinstance(usage, dict):
        keys = ("input_tokens", "output_tokens", "total_tokens")
        if all(type(usage.get(key)) is int and usage[key] >= 0 for key in keys):
            usage_metadata = {key: usage[key] for key in keys}
    return AIMessage(
        content="".join(text),
        tool_calls=calls,
        additional_kwargs={RESPONSES_OUTPUT_KEY: deepcopy(output)},
        response_metadata={
            "response_id": response.get("id"),
            "model_name": response.get("model"),
            "finish_reason": "tool_calls" if calls else "stop",
        },
        usage_metadata=usage_metadata,
    )


class ChatGPTResponsesModel(BaseChatModel):
    """Use ainvoke with an account-specific OAuth getter and caller-owned client."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str | None = None
    access_token_getter: AccessTokenGetter = Field(exclude=True, repr=False)
    http_client: httpx.AsyncClient = Field(exclude=True, repr=False)
    timeout_seconds: float = Field(default=180.0, gt=0)
    cache: Literal[False] = False
    disable_streaming: Literal[True] = True

    @property
    def _llm_type(self) -> str:
        return "chatgpt-plan-responses"

    @property
    def _identifying_params(self) -> JsonObject:
        return {"model": self.model}

    def get_num_tokens(self, text: str) -> int:
        """Estimate tokens locally without downloading a tokenizer."""
        return (len(text) + 3) // 4

    def get_num_tokens_from_messages(
        self, messages: list[BaseMessage], tools: Sequence[Any] | None = None
    ) -> int:
        """Return a local estimate, including supplied tool schemas."""
        return count_tokens_approximately(
            messages, tools=_tool_specs(tools) if tools else None
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Use ainvoke or an async DeepAgents graph with this model.")

    async def _token(self) -> str:
        token = await self.access_token_getter()
        if not isinstance(token, str) or not token or any(char.isspace() for char in token):
            raise ResponsesError("No valid ChatGPT access token is available. Sign in again.")
        return token

    async def _catalog(self, token: str) -> list[str]:
        try:
            response = await self.http_client.get(
                _MODELS_URL,
                headers={"Authorization": "Bearer " + token},
                follow_redirects=False,
                timeout=self.timeout_seconds,
            )
            _check_status(response)
            payload = response.json()
        except httpx.HTTPError:
            raise ResponsesError("The model catalog request could not be completed.") from None
        except ValueError:
            raise ResponsesError("The model catalog is not valid JSON.") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ResponsesError("The model catalog is missing its models array.")
        models: list[str] = []
        for entry in payload["models"]:
            if not isinstance(entry, dict):
                raise ResponsesError("The model catalog contains an invalid entry.")
            if entry.get("visibility") == "list":
                slug = entry.get("slug")
                if not isinstance(slug, str) or not slug:
                    raise ResponsesError("The model catalog contains an invalid model slug.")
                if slug not in models:
                    models.append(slug)
        return models

    async def available_models(self) -> list[str]:
        """Return visible account-specific slugs in server order without selecting one."""
        return await self._catalog(await self._token())

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        unknown = set(kwargs) - (_ALLOWED_OPTIONS - {"tools", "tool_choice"})
        if unknown:
            raise ValueError("Unsupported ChatGPT plan options: " + ", ".join(sorted(unknown)))
        options = {**kwargs, "tools": _tool_specs(tools, strict)}
        if tool_choice is not None:
            options["tool_choice"] = tool_choice
        return self.bind(**options)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop is not None:
            raise ValueError("Stop sequences are not supported by this adapter.")
        unknown = set(kwargs) - _ALLOWED_OPTIONS
        if unknown:
            raise ValueError("Unsupported ChatGPT plan options: " + ", ".join(sorted(unknown)))
        if not self.model:
            raise ValueError("Select a model from available_models() before inference.")
        options = deepcopy(kwargs)
        specs = _tool_specs(options.pop("tools", []))
        kinds = {spec["name"]: spec["type"] for spec in specs}
        if "tool_choice" in options:
            choice = options["tool_choice"]
            if choice is True or choice == "any":
                choice = "required"
            elif choice is False:
                choice = "none"
            if not isinstance(choice, str) or choice not in {"auto", "none", "required"}:
                raise ValueError("tool_choice must be auto, none, required, or any.")
            if choice == "required" and not specs:
                raise ValueError("A required tool choice needs at least one tool.")
            options["tool_choice"] = choice
        body = {
            **options,
            "model": self.model,
            "input": _history(messages),
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        if specs:
            body["tools"] = [
                {
                    "type": "namespace",
                    "name": _NAMESPACE,
                    "description": "Tools executed by the local Taskmate agent.",
                    "tools": specs,
                }
            ]
        token = await self._token()
        if self.model not in await self._catalog(token):
            raise ValueError("The selected model is not in this ChatGPT account's visible catalog.")
        completed_items: dict[int, JsonObject] = {}
        try:
            async with self.http_client.stream(
                "POST",
                _RESPONSES_URL,
                headers={"Authorization": "Bearer " + token, "Accept": "text/event-stream"},
                json=body,
                follow_redirects=False,
                timeout=self.timeout_seconds,
            ) as stream:
                _check_status(stream)
                content_type = stream.headers.get("content-type", "").split(";", 1)[0].strip()
                if content_type != "text/event-stream":
                    raise ResponsesError("Expected an event stream from the Responses endpoint.")
                async for event in _events(stream):
                    kind = event["type"]
                    if kind in {"error", "response.failed", "response.incomplete", "response.cancelled"}:
                        raise ResponsesError("The inference stream reported an unsuccessful response.")
                    if kind == "response.output_text.delta" and run_manager is not None:
                        delta = event.get("delta")
                        if isinstance(delta, str):
                            await run_manager.on_llm_new_token(delta)
                    if kind == "response.output_item.done":
                        index, item = event.get("output_index"), event.get("item")
                        if type(index) is not int or index < 0 or not isinstance(item, dict):
                            raise ResponsesError("The inference stream contains an invalid completed item.")
                        completed_items[index] = item
                    if kind == "response.completed":
                        response = event.get("response")
                        if not isinstance(response, dict) or response.get("status") != "completed":
                            raise ResponsesError("The inference stream has an invalid completion.")
                        response = deepcopy(response)
                        output = response.get("output")
                        if not isinstance(output, list):
                            raise ResponsesError("The completed response is missing its output items.")
                        for index, item in completed_items.items():
                            if index >= len(output) or not isinstance(output[index], dict):
                                raise ResponsesError("The completed output does not match the stream.")
                            if item.get("id") != output[index].get("id"):
                                raise ResponsesError("The completed output does not match the stream.")
                            output[index] = {**output[index], **item}
                        message = _message(response, kinds)
                        return ChatResult(generations=[ChatGeneration(message=message)])
        except httpx.HTTPError:
            raise ResponsesError("The inference connection failed.") from None
        raise ResponsesError("The inference stream ended before a completed response.")

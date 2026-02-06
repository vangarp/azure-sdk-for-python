# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for wire format models."""
import json
import pytest

from azure.ai.agentserver.core.wire.models import (
    AgentInfo,
    AgentRequest,
    AgentResponse,
    AgentStreamEvent,
    ContentPart,
    ErrorInfo,
    InputMessage,
    OutputItem,
    OutputItemType,
    StreamEventType,
    ToolCallInfo,
)


class TestOutputItem:
    """Tests for OutputItem dataclass."""

    def test_text_message_roundtrip(self):
        item = OutputItem(
            type=OutputItemType.TEXT_MESSAGE,
            role="assistant",
            content="Hello, world!",
        )
        d = item.to_dict()
        assert d["type"] == "text_message"
        assert d["role"] == "assistant"
        assert d["content"] == "Hello, world!"

        restored = OutputItem.from_dict(d)
        assert restored.type == OutputItemType.TEXT_MESSAGE
        assert restored.role == "assistant"
        assert restored.content == "Hello, world!"

    def test_function_call_roundtrip(self):
        item = OutputItem(
            type=OutputItemType.FUNCTION_CALL,
            call_id="call_123",
            name="get_weather",
            arguments='{"city": "Seattle"}',
        )
        d = item.to_dict()
        assert d["type"] == "function_call"
        assert d["call_id"] == "call_123"

        restored = OutputItem.from_dict(d)
        assert restored.type == OutputItemType.FUNCTION_CALL
        assert restored.name == "get_weather"
        assert restored.arguments == '{"city": "Seattle"}'

    def test_function_call_output_roundtrip(self):
        item = OutputItem(
            type=OutputItemType.FUNCTION_CALL_OUTPUT,
            call_id="call_123",
            output="72°F, Sunny",
        )
        d = item.to_dict()
        assert d["type"] == "function_call_output"

        restored = OutputItem.from_dict(d)
        assert restored.call_id == "call_123"
        assert restored.output == "72°F, Sunny"

    def test_json_serialization(self):
        item = OutputItem(
            type=OutputItemType.TEXT_MESSAGE,
            role="assistant",
            content="test",
        )
        d = item.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["type"] == "text_message"

        restored = OutputItem.from_dict(parsed)
        assert restored.content == "test"


class TestInputMessage:
    """Tests for InputMessage dataclass."""

    def test_user_message(self):
        msg = InputMessage(role="user", content="What's the weather?")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "What's the weather?"

    def test_message_with_tool_calls(self):
        msg = InputMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCallInfo(id="tc_1", name="get_weather", arguments='{"city":"NYC"}'),
            ],
        )
        d = msg.to_dict()
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["name"] == "get_weather"

        restored = InputMessage.from_dict(d)
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].id == "tc_1"

    def test_tool_message(self):
        msg = InputMessage(role="tool", content="72°F", tool_call_id="tc_1")
        d = msg.to_dict()
        assert d["tool_call_id"] == "tc_1"

    def test_content_parts(self):
        msg = InputMessage(
            role="user",
            content_parts=[
                ContentPart(type="text", text="Hello"),
                ContentPart(type="image", extra={"url": "https://example.com/img.png"}),
            ],
        )
        d = msg.to_dict()
        assert len(d["content_parts"]) == 2
        restored = InputMessage.from_dict(d)
        assert restored.content_parts[0].text == "Hello"
        assert restored.content_parts[1].type == "image"


class TestAgentRequest:
    """Tests for AgentRequest dataclass."""

    def test_minimal_request(self):
        req = AgentRequest(
            response_id="resp_1",
            conversation_id="conv_1",
            stream=False,
            messages=[InputMessage(role="user", content="Hi")],
        )
        d = req.to_dict()
        assert d["response_id"] == "resp_1"
        assert d["stream"] is False
        assert len(d["messages"]) == 1

    def test_full_request_roundtrip(self):
        req = AgentRequest(
            response_id="resp_2",
            conversation_id="conv_2",
            stream=True,
            instructions="Be helpful",
            messages=[
                InputMessage(role="user", content="Tell me a joke"),
            ],
            metadata={"key": "value"},
            agent=AgentInfo(name="TestBot", version="1.0"),
            extensions={"custom_field": 42},
        )
        json_str = req.to_json()
        restored = AgentRequest.from_json(json_str)
        assert restored.response_id == "resp_2"
        assert restored.stream is True
        assert restored.instructions == "Be helpful"
        assert restored.metadata == {"key": "value"}
        assert restored.agent.name == "TestBot"
        assert restored.extensions["custom_field"] == 42


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    def test_success_response(self):
        resp = AgentResponse(
            status="completed",
            output=[
                OutputItem(
                    type=OutputItemType.TEXT_MESSAGE,
                    role="assistant",
                    content="Here's a joke...",
                ),
            ],
        )
        d = resp.to_dict()
        assert d["status"] == "completed"
        assert len(d["output"]) == 1

        restored = AgentResponse.from_dict(d)
        assert restored.output[0].content == "Here's a joke..."

    def test_error_response(self):
        resp = AgentResponse(
            status="failed",
            error=ErrorInfo(code="rate_limit", message="Too many requests"),
        )
        d = resp.to_dict()
        assert d["error"]["code"] == "rate_limit"

        restored = AgentResponse.from_dict(d)
        assert restored.error.message == "Too many requests"


class TestAgentStreamEvent:
    """Tests for AgentStreamEvent dataclass."""

    def test_text_delta(self):
        event = AgentStreamEvent(
            type=StreamEventType.TEXT_DELTA,
            output_index=0,
            delta="Hello",
        )
        d = event.to_dict()
        assert d["type"] == "text_delta"
        assert d["delta"] == "Hello"

    def test_output_item_added(self):
        item = OutputItem(
            type=OutputItemType.TEXT_MESSAGE, role="assistant", content=""
        )
        event = AgentStreamEvent(
            type=StreamEventType.OUTPUT_ITEM_ADDED,
            output_index=0,
            item=item,
        )
        d = event.to_dict()
        assert d["type"] == "output_item_added"
        assert d["item"]["type"] == "text_message"

    def test_completed(self):
        resp = AgentResponse(status="completed", output=[])
        event = AgentStreamEvent(
            type=StreamEventType.COMPLETED,
            output_index=0,
            response=resp,
        )
        d = event.to_dict()
        assert d["type"] == "completed"

    def test_error_event(self):
        event = AgentStreamEvent(
            type=StreamEventType.ERROR,
            error=ErrorInfo(code="internal", message="Something broke"),
        )
        d = event.to_dict()
        assert d["type"] == "error"
        assert d["error"]["code"] == "internal"

    def test_function_call_arguments_delta(self):
        event = AgentStreamEvent(
            type=StreamEventType.FUNCTION_CALL_ARGUMENTS_DELTA,
            output_index=1,
            delta='{"city":',
        )
        d = event.to_dict()
        assert d["type"] == "function_call_arguments_delta"


class TestStreamEventType:
    """Tests for StreamEventType enum."""

    def test_all_values(self):
        assert StreamEventType.TEXT_DELTA.value == "text_delta"
        assert StreamEventType.FUNCTION_CALL_ARGUMENTS_DELTA.value == "function_call_arguments_delta"
        assert StreamEventType.OUTPUT_ITEM_ADDED.value == "output_item_added"
        assert StreamEventType.OUTPUT_ITEM_DONE.value == "output_item_done"
        assert StreamEventType.COMPLETED.value == "completed"
        assert StreamEventType.ERROR.value == "error"


class TestOutputItemType:
    """Tests for OutputItemType enum."""

    def test_all_values(self):
        assert OutputItemType.TEXT_MESSAGE.value == "text_message"
        assert OutputItemType.FUNCTION_CALL.value == "function_call"
        assert OutputItemType.FUNCTION_CALL_OUTPUT.value == "function_call_output"

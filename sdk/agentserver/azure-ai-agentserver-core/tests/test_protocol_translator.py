# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for the protocol translator.

These tests validate the conversion logic between OpenAI Responses API
format and the simplified wire format, without requiring full OpenAI SDK
installations (we mock the AgentRunContext).
"""
import json
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional
from unittest.mock import MagicMock, PropertyMock

import pytest

from azure.ai.agentserver.core.wire.models import (
    AgentRequest,
    AgentResponse,
    AgentStreamEvent,
    ErrorInfo,
    InputMessage,
    OutputItem,
    OutputItemType,
    StreamEventType,
)
from azure.ai.agentserver.core.server.protocol_translator import (
    _convert_input_to_messages,
    request_to_wire,
    wire_response_to_openai,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(
    payload: dict,
    response_id: str = "resp_test",
    conversation_id: str = "conv_test",
    stream: bool = False,
):
    """Build a minimal mock of AgentRunContext for testing."""
    ctx = MagicMock()
    ctx.response_id = response_id
    ctx.conversation_id = conversation_id
    ctx.stream = stream

    # request behaves like a dict-like object with .get()
    request_mock = MagicMock()
    request_mock.get = lambda key, default=None: payload.get(key, default)
    ctx.request = request_mock

    return ctx


# ---------------------------------------------------------------------------
# request_to_wire tests
# ---------------------------------------------------------------------------


class TestRequestToWire:
    """Tests for request_to_wire conversion."""

    def test_simple_string_input(self):
        ctx = _make_context({"input": "Hello"}, response_id="r1", conversation_id="c1")
        wire = request_to_wire(ctx)
        assert isinstance(wire, AgentRequest)
        assert wire.response_id == "r1"
        assert wire.conversation_id == "c1"
        assert wire.stream is False
        assert len(wire.messages) == 1
        assert wire.messages[0].role == "user"
        assert wire.messages[0].content == "Hello"

    def test_instructions_preserved(self):
        ctx = _make_context(
            {"input": "Hi", "instructions": "Be concise"},
            stream=True,
        )
        wire = request_to_wire(ctx)
        assert wire.instructions == "Be concise"
        assert wire.stream is True

    def test_no_input(self):
        ctx = _make_context({})
        wire = request_to_wire(ctx)
        assert wire.messages == []

    def test_message_input_items(self):
        ctx = _make_context(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "What is 2+2?"}],
                    },
                ]
            }
        )
        wire = request_to_wire(ctx)
        assert len(wire.messages) == 1
        assert wire.messages[0].role == "user"

    def test_function_call_output_input(self):
        ctx = _make_context(
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_abc",
                        "output": "42",
                    },
                ]
            }
        )
        wire = request_to_wire(ctx)
        assert len(wire.messages) == 1
        msg = wire.messages[0]
        assert msg.role == "tool"
        assert msg.content == "42"
        assert msg.tool_call_id == "call_abc"


class TestConvertInputToMessages:
    """Tests for _convert_input_to_messages helper."""

    def test_none_input(self):
        request = MagicMock()
        request.get = lambda key, default=None: None
        msgs = _convert_input_to_messages(request)
        assert msgs == []

    def test_string_input(self):
        request = MagicMock()
        request.get = lambda key, default=None: "test" if key == "input" else default
        msgs = _convert_input_to_messages(request)
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "test"


# ---------------------------------------------------------------------------
# wire_response_to_openai tests
# ---------------------------------------------------------------------------


class TestWireResponseToOpenai:
    """Tests for wire_response_to_openai conversion."""

    def test_simple_text_response(self):
        wire_resp = AgentResponse(
            status="completed",
            output=[
                OutputItem(
                    type=OutputItemType.TEXT_MESSAGE,
                    role="assistant",
                    content="Hello!",
                ),
            ],
        )
        ctx = _make_context({}, response_id="resp_1", conversation_id="conv_1")
        openai_resp = wire_response_to_openai(wire_resp, ctx)

        # Should return an OpenAI Response dict-like object
        resp_dict = openai_resp.as_dict() if hasattr(openai_resp, "as_dict") else openai_resp
        assert resp_dict["id"] == "resp_1"
        assert resp_dict["status"] == "completed"
        assert len(resp_dict["output"]) == 1

    def test_function_call_response(self):
        wire_resp = AgentResponse(
            status="completed",
            output=[
                OutputItem(
                    type=OutputItemType.FUNCTION_CALL,
                    call_id="call_xyz",
                    name="get_weather",
                    arguments='{"city": "Seattle"}',
                ),
            ],
        )
        ctx = _make_context({}, response_id="resp_fc", conversation_id="conv_1")
        openai_resp = wire_response_to_openai(wire_resp, ctx)
        resp_dict = openai_resp.as_dict() if hasattr(openai_resp, "as_dict") else openai_resp
        assert len(resp_dict["output"]) == 1
        item = resp_dict["output"][0]
        assert item["type"] == "function_call"
        assert item["name"] == "get_weather"

    def test_error_response(self):
        wire_resp = AgentResponse(
            status="failed",
            error=ErrorInfo(code="rate_limit", message="Slow down"),
        )
        ctx = _make_context({}, response_id="resp_err", conversation_id="conv_1")
        openai_resp = wire_response_to_openai(wire_resp, ctx)
        resp_dict = openai_resp.as_dict() if hasattr(openai_resp, "as_dict") else openai_resp
        assert resp_dict["status"] == "failed"

    def test_empty_output(self):
        wire_resp = AgentResponse(status="completed", output=[])
        ctx = _make_context({}, response_id="resp_empty", conversation_id="conv_1")
        openai_resp = wire_response_to_openai(wire_resp, ctx)
        resp_dict = openai_resp.as_dict() if hasattr(openai_resp, "as_dict") else openai_resp
        assert resp_dict["output"] == []

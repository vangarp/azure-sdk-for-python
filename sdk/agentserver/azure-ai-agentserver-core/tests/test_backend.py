# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Unit tests for the backend client abstraction and adapters."""
import asyncio
from typing import AsyncIterator

import pytest

from azure.ai.agentserver.core.server.backend import BackendClient
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


class MockBackend(BackendClient):
    """A simple in-memory backend for testing."""

    def __init__(self, response: AgentResponse = None, events: list = None):
        self._response = response or AgentResponse(status="completed", output=[])
        self._events = events or []
        self._closed = False

    async def run(self, request: AgentRequest) -> AgentResponse:
        return self._response

    async def run_stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
        for event in self._events:
            yield event

    async def close(self) -> None:
        self._closed = True


def _make_request(**kwargs) -> AgentRequest:
    defaults = {
        "response_id": "resp_1",
        "conversation_id": "conv_1",
        "stream": False,
        "messages": [InputMessage(role="user", content="hello")],
    }
    defaults.update(kwargs)
    return AgentRequest(**defaults)


class TestMockBackend:
    """Test the mock backend to validate the ABC contract."""

    @pytest.mark.asyncio
    async def test_run_returns_response(self):
        resp = AgentResponse(
            status="completed",
            output=[
                OutputItem(
                    type=OutputItemType.TEXT_MESSAGE,
                    role="assistant",
                    content="Hi there!",
                )
            ],
        )
        backend = MockBackend(response=resp)
        result = await backend.run(_make_request())
        assert result.status == "completed"
        assert result.output[0].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_run_stream_yields_events(self):
        events = [
            AgentStreamEvent(
                type=StreamEventType.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=OutputItem(type=OutputItemType.TEXT_MESSAGE, role="assistant", content=""),
            ),
            AgentStreamEvent(
                type=StreamEventType.TEXT_DELTA,
                output_index=0,
                delta="Hello",
            ),
            AgentStreamEvent(
                type=StreamEventType.OUTPUT_ITEM_DONE,
                output_index=0,
                item=OutputItem(type=OutputItemType.TEXT_MESSAGE, role="assistant", content="Hello"),
            ),
            AgentStreamEvent(type=StreamEventType.COMPLETED, output_index=0),
        ]
        backend = MockBackend(events=events)
        collected = []
        async for event in backend.run_stream(_make_request(stream=True)):
            collected.append(event)
        assert len(collected) == 4
        assert collected[0].type == StreamEventType.OUTPUT_ITEM_ADDED
        assert collected[1].type == StreamEventType.TEXT_DELTA
        assert collected[1].delta == "Hello"
        assert collected[3].type == StreamEventType.COMPLETED

    @pytest.mark.asyncio
    async def test_close(self):
        backend = MockBackend()
        assert not backend._closed
        await backend.close()
        assert backend._closed

    @pytest.mark.asyncio
    async def test_context_manager(self):
        backend = MockBackend()
        async with backend:
            result = await backend.run(_make_request())
            assert result.status == "completed"
        assert backend._closed

    @pytest.mark.asyncio
    async def test_error_response(self):
        resp = AgentResponse(
            status="failed",
            error=ErrorInfo(code="timeout", message="Request timed out"),
        )
        backend = MockBackend(response=resp)
        result = await backend.run(_make_request())
        assert result.status == "failed"
        assert result.error.code == "timeout"

    @pytest.mark.asyncio
    async def test_error_stream_event(self):
        events = [
            AgentStreamEvent(
                type=StreamEventType.ERROR,
                error=ErrorInfo(code="internal", message="boom"),
            ),
        ]
        backend = MockBackend(events=events)
        collected = []
        async for event in backend.run_stream(_make_request(stream=True)):
            collected.append(event)
        assert len(collected) == 1
        assert collected[0].type == StreamEventType.ERROR
        assert collected[0].error.message == "boom"
